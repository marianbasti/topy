"""
Flask application factory and routes for ToPy web interface.
"""
import os
import uuid
import json
import shutil
import tempfile
from typing import Dict, Any, Optional

from flask import (Flask, render_template, request, jsonify, 
                   send_file, redirect, url_for, session)

from .ply_utils import (parse_ply_file, ply_to_voxel_grid, 
                         voxel_grid_to_ply, estimate_grid_resolution,
                         get_ply_info)
from .optimizer import run_optimization, get_job_status, OptimizationJob

# Store active jobs (in production, use Redis or database)
_jobs: Dict[str, OptimizationJob] = {}


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary.
    
    Returns:
        Configured Flask application.
    """
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Default configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'topy-dev-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'topy_uploads')
    app.config['RESULTS_FOLDER'] = os.path.join(tempfile.gettempdir(), 'topy_results')
    
    # Apply custom configuration
    if config:
        app.config.update(config)
    
    # Ensure folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    register_routes(app)
    
    return app


def register_routes(app: Flask) -> None:
    """Register all routes for the application."""
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """
        Handle PLY file upload.
        
        Returns:
            JSON response with file info and suggested parameters.
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.ply'):
            return jsonify({'error': 'Only PLY files are supported'}), 400
        
        # Generate unique job ID
        job_id = uuid.uuid4().hex
        
        # Save uploaded file
        upload_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            f"{job_id}.ply"
        )
        file.save(upload_path)
        
        try:
            # Parse PLY file and get info
            ply_data = parse_ply_file(upload_path)
            suggested_resolution = estimate_grid_resolution(ply_data)
            
            # Store job info
            _jobs[job_id] = OptimizationJob(
                job_id=job_id,
                input_file=upload_path,
                status='uploaded',
                ply_data=ply_data
            )
            
            return jsonify({
                'job_id': job_id,
                'filename': file.filename,
                'num_vertices': ply_data['num_vertices'],
                'num_faces': ply_data['num_faces'],
                'dimensions': ply_data['dimensions'].tolist(),
                'suggested_resolution': list(suggested_resolution),
                'message': 'File uploaded successfully'
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({'error': str(e)}), 400
    
    @app.route('/configure/<job_id>', methods=['POST'])
    def configure_job(job_id: str):
        """
        Configure optimization parameters for a job.
        
        Args:
            job_id: The job identifier.
        
        Returns:
            JSON response confirming configuration.
        """
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = _jobs[job_id]
        
        if job.status not in ('uploaded', 'configured'):
            return jsonify({'error': f'Cannot configure job in {job.status} state'}), 400
        
        try:
            data = request.get_json()
            
            # Extract and validate parameters
            params = {
                'vol_frac': float(data.get('vol_frac', 0.3)),
                'filt_rad': float(data.get('filt_rad', 1.5)),
                'num_iter': int(data.get('num_iter', 50)),
                'p_fac': float(data.get('p_fac', 3.0)),
                'resolution': tuple(data.get('resolution', [20, 20, 20]))
            }
            
            # Validate parameters
            if not 0.01 <= params['vol_frac'] <= 0.99:
                return jsonify({'error': 'Volume fraction must be between 0.01 and 0.99'}), 400
            
            if not 0.5 <= params['filt_rad'] <= 10.0:
                return jsonify({'error': 'Filter radius must be between 0.5 and 10.0'}), 400
            
            if not 1 <= params['num_iter'] <= 500:
                return jsonify({'error': 'Number of iterations must be between 1 and 500'}), 400
            
            if not 1.0 <= params['p_fac'] <= 5.0:
                return jsonify({'error': 'Penalization factor must be between 1.0 and 5.0'}), 400
            
            for dim in params['resolution']:
                if not 3 <= dim <= 100:
                    return jsonify({'error': 'Resolution dimensions must be between 3 and 100'}), 400
            
            job.params = params
            job.status = 'configured'
            
            return jsonify({
                'job_id': job_id,
                'status': 'configured',
                'params': params,
                'message': 'Job configured successfully'
            })
            
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    
    @app.route('/optimize/<job_id>', methods=['POST'])
    def start_optimization(job_id: str):
        """
        Start the optimization process for a configured job.
        
        Args:
            job_id: The job identifier.
        
        Returns:
            JSON response with job status.
        """
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = _jobs[job_id]
        
        if job.status != 'configured':
            return jsonify({'error': f'Job must be configured first. Current status: {job.status}'}), 400
        
        try:
            # Run optimization
            job.status = 'running'
            result_path = run_optimization(job, app.config['RESULTS_FOLDER'])
            
            job.status = 'completed'
            job.result_file = result_path
            
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'message': 'Optimization completed successfully'
            })
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/status/<job_id>', methods=['GET'])
    def job_status(job_id: str):
        """
        Get the current status of a job.
        
        Args:
            job_id: The job identifier.
        
        Returns:
            JSON response with job status.
        """
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = _jobs[job_id]
        
        response = {
            'job_id': job_id,
            'status': job.status,
        }
        
        if job.params:
            response['params'] = job.params
        
        if job.error:
            response['error'] = job.error
        
        if job.progress is not None:
            response['progress'] = job.progress
        
        return jsonify(response)
    
    @app.route('/download/<job_id>', methods=['GET'])
    def download_result(job_id: str):
        """
        Download the optimization result.
        
        Args:
            job_id: The job identifier.
        
        Returns:
            The result file as a download.
        """
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = _jobs[job_id]
        
        if job.status != 'completed':
            return jsonify({'error': f'Job not completed. Status: {job.status}'}), 400
        
        if not job.result_file or not os.path.exists(job.result_file):
            return jsonify({'error': 'Result file not found'}), 404
        
        # Determine file type for download
        file_format = request.args.get('format', 'ply')
        
        if file_format == 'vtk' and job.result_file.endswith('.vtk'):
            return send_file(
                job.result_file,
                as_attachment=True,
                download_name=f"optimized_{job_id}.vtk"
            )
        elif file_format == 'ply':
            # Convert VTK to PLY if needed
            if job.result_file.endswith('.vtk'):
                # Use existing PLY result or generate one
                ply_path = job.result_file.replace('.vtk', '.ply')
                if not os.path.exists(ply_path):
                    # Generate PLY from the optimization result
                    if job.result_grid is None:
                        return jsonify({'error': 'Result grid not available'}), 500
                    ply_path = voxel_grid_to_ply(
                        job.result_grid,
                        threshold=0.5,
                        output_path=ply_path
                    )
            else:
                ply_path = job.result_file
            
            return send_file(
                ply_path,
                as_attachment=True,
                download_name=f"optimized_{job_id}.ply"
            )
        else:
            return jsonify({'error': 'Unsupported format'}), 400
    
    @app.route('/cleanup/<job_id>', methods=['DELETE'])
    def cleanup_job(job_id: str):
        """
        Clean up a job and its associated files.
        
        Args:
            job_id: The job identifier.
        
        Returns:
            JSON response confirming cleanup.
        """
        if job_id not in _jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = _jobs[job_id]
        
        # Remove files
        if job.input_file and os.path.exists(job.input_file):
            os.remove(job.input_file)
        
        if job.result_file and os.path.exists(job.result_file):
            os.remove(job.result_file)
            # Also remove PLY version if exists
            ply_path = job.result_file.replace('.vtk', '.ply')
            if os.path.exists(ply_path):
                os.remove(ply_path)
        
        # Remove from jobs dict
        del _jobs[job_id]
        
        return jsonify({
            'job_id': job_id,
            'message': 'Job cleaned up successfully'
        })
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'healthy'})


def get_jobs() -> Dict[str, OptimizationJob]:
    """Get the current jobs dictionary (for testing)."""
    return _jobs


def clear_jobs() -> None:
    """Clear all jobs (for testing)."""
    _jobs.clear()
