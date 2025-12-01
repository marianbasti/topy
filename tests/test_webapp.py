"""Tests for the ToPy web application."""
import os
import json
import tempfile
import pytest
import numpy as np

from topy.webapp.app import create_app, get_jobs, clear_jobs
from topy.webapp.ply_utils import (
    parse_ply_file, 
    ply_to_voxel_grid, 
    voxel_grid_to_ply,
    estimate_grid_resolution
)


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app({
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
    })
    yield app
    clear_jobs()


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_ply_file():
    """Create a sample PLY file for testing."""
    # Create a simple cube mesh
    ply_content = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
3 0 1 2
3 0 2 3
3 4 6 5
3 4 7 6
3 0 4 5
3 0 5 1
3 2 6 7
3 2 7 3
3 0 3 7
3 0 7 4
3 1 5 6
3 1 6 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as f:
        f.write(ply_content)
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test that health check returns healthy status."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'


class TestIndex:
    """Test index page."""
    
    def test_index_returns_html(self, client):
        """Test that index returns HTML page."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'ToPy' in response.data


class TestUpload:
    """Test file upload functionality."""
    
    def test_upload_no_file(self, client):
        """Test upload with no file returns error."""
        response = client.post('/upload')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_upload_empty_filename(self, client):
        """Test upload with empty filename returns error."""
        response = client.post('/upload', data={
            'file': (b'', '')
        })
        assert response.status_code == 400
    
    def test_upload_wrong_extension(self, client):
        """Test upload with wrong file extension returns error."""
        import io
        response = client.post('/upload', data={
            'file': (io.BytesIO(b'test content'), 'test.txt')
        }, content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'PLY' in data['error']
    
    def test_upload_valid_ply(self, client, sample_ply_file):
        """Test upload with valid PLY file succeeds."""
        with open(sample_ply_file, 'rb') as f:
            response = client.post('/upload', data={
                'file': (f, 'test.ply')
            }, content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'job_id' in data
        assert data['num_vertices'] == 8
        assert data['num_faces'] == 12


class TestConfigure:
    """Test configuration endpoint."""
    
    def test_configure_nonexistent_job(self, client):
        """Test configuring non-existent job returns error."""
        response = client.post('/configure/nonexistent', 
                               data=json.dumps({'vol_frac': 0.3}),
                               content_type='application/json')
        assert response.status_code == 404
    
    def test_configure_valid_job(self, client, sample_ply_file):
        """Test configuring a valid job succeeds."""
        # First upload
        with open(sample_ply_file, 'rb') as f:
            upload_response = client.post('/upload', data={
                'file': (f, 'test.ply')
            }, content_type='multipart/form-data')
        
        job_id = json.loads(upload_response.data)['job_id']
        
        # Then configure
        config_response = client.post(f'/configure/{job_id}',
                                      data=json.dumps({
                                          'vol_frac': 0.3,
                                          'filt_rad': 1.5,
                                          'num_iter': 10,
                                          'p_fac': 3.0,
                                          'resolution': [5, 5, 5]
                                      }),
                                      content_type='application/json')
        
        assert config_response.status_code == 200
        data = json.loads(config_response.data)
        assert data['status'] == 'configured'
    
    def test_configure_invalid_vol_frac(self, client, sample_ply_file):
        """Test configuring with invalid volume fraction returns error."""
        # First upload
        with open(sample_ply_file, 'rb') as f:
            upload_response = client.post('/upload', data={
                'file': (f, 'test.ply')
            }, content_type='multipart/form-data')
        
        job_id = json.loads(upload_response.data)['job_id']
        
        # Configure with invalid vol_frac
        config_response = client.post(f'/configure/{job_id}',
                                      data=json.dumps({
                                          'vol_frac': 1.5  # Invalid
                                      }),
                                      content_type='application/json')
        
        assert config_response.status_code == 400


class TestStatus:
    """Test status endpoint."""
    
    def test_status_nonexistent_job(self, client):
        """Test status of non-existent job returns error."""
        response = client.get('/status/nonexistent')
        assert response.status_code == 404
    
    def test_status_valid_job(self, client, sample_ply_file):
        """Test status of valid job returns correct status."""
        # First upload
        with open(sample_ply_file, 'rb') as f:
            upload_response = client.post('/upload', data={
                'file': (f, 'test.ply')
            }, content_type='multipart/form-data')
        
        job_id = json.loads(upload_response.data)['job_id']
        
        # Check status
        status_response = client.get(f'/status/{job_id}')
        
        assert status_response.status_code == 200
        data = json.loads(status_response.data)
        assert data['status'] == 'uploaded'


class TestPlyUtils:
    """Test PLY utility functions."""
    
    def test_parse_ply_file(self, sample_ply_file):
        """Test parsing a PLY file."""
        data = parse_ply_file(sample_ply_file)
        
        assert data['num_vertices'] == 8
        assert data['num_faces'] == 12
        assert data['vertices'].shape == (8, 3)
        assert len(data['dimensions']) == 3
    
    def test_ply_to_voxel_grid(self, sample_ply_file):
        """Test converting PLY to voxel grid."""
        ply_data = parse_ply_file(sample_ply_file)
        grid = ply_to_voxel_grid(ply_data, (5, 5, 5))
        
        assert grid.shape == (5, 5, 5)
        assert grid.max() <= 1.0
        assert grid.min() >= 0.0
    
    def test_voxel_grid_to_ply(self):
        """Test converting voxel grid back to PLY."""
        # Create a simple grid
        grid = np.zeros((3, 3, 3))
        grid[1, 1, 1] = 1.0  # Single solid voxel
        
        output_path = voxel_grid_to_ply(grid, threshold=0.5)
        
        assert os.path.exists(output_path)
        
        # Verify it's a valid PLY file
        data = parse_ply_file(output_path)
        assert data['num_vertices'] == 8  # One cube has 8 vertices
        
        # Cleanup
        os.remove(output_path)
    
    def test_estimate_grid_resolution(self, sample_ply_file):
        """Test grid resolution estimation."""
        ply_data = parse_ply_file(sample_ply_file)
        resolution = estimate_grid_resolution(ply_data, target_elements=1000)
        
        assert len(resolution) == 3
        assert all(3 <= r <= 100 for r in resolution)


class TestCleanup:
    """Test cleanup endpoint."""
    
    def test_cleanup_nonexistent_job(self, client):
        """Test cleanup of non-existent job returns error."""
        response = client.delete('/cleanup/nonexistent')
        assert response.status_code == 404
    
    def test_cleanup_valid_job(self, client, sample_ply_file):
        """Test cleanup of valid job succeeds."""
        # First upload
        with open(sample_ply_file, 'rb') as f:
            upload_response = client.post('/upload', data={
                'file': (f, 'test.ply')
            }, content_type='multipart/form-data')
        
        job_id = json.loads(upload_response.data)['job_id']
        
        # Cleanup
        cleanup_response = client.delete(f'/cleanup/{job_id}')
        
        assert cleanup_response.status_code == 200
        
        # Verify job is gone
        status_response = client.get(f'/status/{job_id}')
        assert status_response.status_code == 404
