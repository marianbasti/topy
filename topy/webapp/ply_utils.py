"""
Utility functions for PLY file handling and mesh conversion.
"""
import os
import uuid
import tempfile
from typing import Dict, Any, Optional, Tuple

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    PlyData = None
    PlyElement = None


def parse_ply_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a PLY file and extract mesh information.
    
    Args:
        filepath: Path to the PLY file.
    
    Returns:
        Dictionary containing vertices, faces, and bounding box information.
    
    Raises:
        ImportError: If plyfile module is not installed.
        ValueError: If the file format is invalid.
    """
    if PlyData is None:
        raise ImportError("plyfile module is required for PLY file support. "
                          "Install it with: pip install plyfile")
    
    plydata = PlyData.read(filepath)
    
    # Extract vertices
    vertex_data = plydata['vertex']
    vertices = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ])
    
    # Extract faces if available
    faces = None
    if 'face' in plydata:
        face_data = plydata['face']
        faces = np.array([f[0] for f in face_data['vertex_indices']])
    
    # Calculate bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    dimensions = max_coords - min_coords
    
    return {
        'vertices': vertices,
        'faces': faces,
        'num_vertices': len(vertices),
        'num_faces': len(faces) if faces is not None else 0,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'dimensions': dimensions,
        'center': (min_coords + max_coords) / 2
    }


def ply_to_voxel_grid(ply_data: Dict[str, Any], 
                       resolution: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert PLY mesh data to a voxel grid for topology optimization.
    
    Args:
        ply_data: Parsed PLY data from parse_ply_file.
        resolution: Tuple (nx, ny, nz) specifying grid resolution.
    
    Returns:
        3D numpy array representing the voxel grid (1 = solid, 0 = void).
    """
    vertices = ply_data['vertices']
    min_coords = ply_data['min_coords']
    dimensions = ply_data['dimensions']
    
    nx, ny, nz = resolution
    
    # Normalize vertices to [0, 1] range
    normalized = (vertices - min_coords) / dimensions
    
    # Scale to grid indices
    indices = np.floor(normalized * np.array([nx, ny, nz])).astype(int)
    
    # Clamp indices to valid range
    indices = np.clip(indices, 0, np.array([nx-1, ny-1, nz-1]))
    
    # Create voxel grid
    grid = np.zeros((nz, ny, nx), dtype=float)
    
    # Mark voxels containing vertices as solid
    for idx in indices:
        grid[idx[2], idx[1], idx[0]] = 1.0
    
    return grid


def voxel_grid_to_ply(grid: np.ndarray, 
                       threshold: float = 0.5,
                       output_path: Optional[str] = None) -> str:
    """
    Convert a voxel grid (from optimization result) back to PLY format.
    
    Args:
        grid: 3D numpy array of density values.
        threshold: Density threshold for including voxels (default 0.5).
        output_path: Optional path for output file. If None, generates temp file.
    
    Returns:
        Path to the generated PLY file.
    """
    if PlyData is None:
        raise ImportError("plyfile module is required for PLY file support.")
    
    # Find voxels above threshold
    nz, ny, nx = grid.shape
    solid_voxels = np.argwhere(grid > threshold)
    
    if len(solid_voxels) == 0:
        raise ValueError("No voxels above threshold found in the grid.")
    
    # Generate vertices and faces for each solid voxel (as cubes)
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Define cube vertices relative to origin
    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)
    
    # Define cube faces (quads as triangles)
    cube_faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ]
    
    for voxel in solid_voxels:
        z, y, x = voxel
        
        # Translate cube vertices to voxel position
        verts = cube_vertices + np.array([x, y, z])
        all_vertices.extend(verts.tolist())
        
        # Add faces with offset
        for face in cube_faces:
            all_faces.append([f + vertex_offset for f in face])
        vertex_offset += 8
    
    # Convert to numpy structured arrays (must be 1D for plyfile)
    # Create vertices as list of tuples
    vertices_tuples = [(v[0], v[1], v[2]) for v in all_vertices]
    vertices_array = np.array(vertices_tuples, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
    ])
    
    faces_array = np.empty(len(all_faces), dtype=[
        ('vertex_indices', 'i4', (3,))
    ])
    faces_array['vertex_indices'] = all_faces
    
    # Create PLY elements
    vertex_element = PlyElement.describe(vertices_array, 'vertex')
    face_element = PlyElement.describe(faces_array, 'face')
    
    # Create and save PLY file
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), 
                                    f"topy_result_{uuid.uuid4().hex[:8]}.ply")
    
    plydata = PlyData([vertex_element, face_element], text=True)
    plydata.write(output_path)
    
    return output_path


def estimate_grid_resolution(ply_data: Dict[str, Any], 
                              target_elements: int = 10000) -> Tuple[int, int, int]:
    """
    Estimate appropriate grid resolution based on mesh dimensions and target element count.
    
    Args:
        ply_data: Parsed PLY data.
        target_elements: Approximate target number of elements.
    
    Returns:
        Tuple (nx, ny, nz) of grid dimensions.
    """
    dimensions = ply_data['dimensions']
    
    # Normalize dimensions
    max_dim = dimensions.max()
    if max_dim == 0:
        return (10, 10, 10)
    
    normalized = dimensions / max_dim
    
    # Calculate base resolution
    base_res = int(np.cbrt(target_elements / np.prod(normalized)))
    base_res = max(5, min(100, base_res))  # Clamp between 5 and 100
    
    # Scale by relative dimensions
    nx = max(3, int(base_res * normalized[0]))
    ny = max(3, int(base_res * normalized[1]))
    nz = max(3, int(base_res * normalized[2]))
    
    return (nx, ny, nz)


def get_ply_info(filepath: str) -> Dict[str, Any]:
    """
    Get basic information about a PLY file without full parsing.
    
    Args:
        filepath: Path to the PLY file.
    
    Returns:
        Dictionary with file information.
    """
    if PlyData is None:
        raise ImportError("plyfile module is required for PLY file support.")
    
    plydata = PlyData.read(filepath)
    
    info = {
        'format': 'ascii' if plydata.text else 'binary',
        'elements': {}
    }
    
    for element in plydata.elements:
        info['elements'][element.name] = {
            'count': len(element.data),
            'properties': [prop.name for prop in element.properties]
        }
    
    return info
