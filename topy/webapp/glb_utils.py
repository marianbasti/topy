"""
Utility functions for GLB (glTF Binary) file handling and mesh conversion.
"""
import os
import struct
import json
import uuid
import tempfile
from typing import Dict, Any, Optional, Tuple

import numpy as np


def parse_glb_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a GLB file and extract mesh information.
    
    Args:
        filepath: Path to the GLB file.
    
    Returns:
        Dictionary containing vertices, faces, and bounding box information.
    
    Raises:
        ValueError: If the file format is invalid.
    """
    with open(filepath, 'rb') as f:
        # Read GLB header
        magic = f.read(4)
        if magic != b'glTF':
            raise ValueError("Invalid GLB file: missing glTF magic number")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version != 2:
            raise ValueError(f"Unsupported glTF version: {version}")
        
        length = struct.unpack('<I', f.read(4))[0]
        
        # Read JSON chunk
        json_chunk_length = struct.unpack('<I', f.read(4))[0]
        json_chunk_type = f.read(4)
        if json_chunk_type != b'JSON':
            raise ValueError("Invalid GLB file: missing JSON chunk")
        
        json_data = json.loads(f.read(json_chunk_length).decode('utf-8'))
        
        # Read binary chunk
        bin_chunk_length = struct.unpack('<I', f.read(4))[0]
        bin_chunk_type = f.read(4)
        if bin_chunk_type != b'BIN\x00':
            raise ValueError("Invalid GLB file: missing BIN chunk")
        
        bin_data = f.read(bin_chunk_length)
    
    # Extract mesh data from the first mesh/primitive
    if 'meshes' not in json_data or len(json_data['meshes']) == 0:
        raise ValueError("No meshes found in GLB file")
    
    mesh = json_data['meshes'][0]
    if 'primitives' not in mesh or len(mesh['primitives']) == 0:
        raise ValueError("No primitives found in mesh")
    
    primitive = mesh['primitives'][0]
    
    # Get accessors and buffer views
    accessors = json_data.get('accessors', [])
    buffer_views = json_data.get('bufferViews', [])
    
    # Extract vertices (POSITION attribute)
    if 'POSITION' not in primitive.get('attributes', {}):
        raise ValueError("No POSITION attribute found")
    
    pos_accessor_idx = primitive['attributes']['POSITION']
    pos_accessor = accessors[pos_accessor_idx]
    pos_buffer_view = buffer_views[pos_accessor['bufferView']]
    
    pos_offset = pos_buffer_view.get('byteOffset', 0) + pos_accessor.get('byteOffset', 0)
    pos_count = pos_accessor['count']
    
    # Read vertex positions (VEC3 of floats)
    vertices = np.frombuffer(
        bin_data[pos_offset:pos_offset + pos_count * 12],
        dtype=np.float32
    ).reshape(-1, 3)
    
    # Extract faces (indices)
    faces = None
    num_faces = 0
    if 'indices' in primitive:
        idx_accessor_idx = primitive['indices']
        idx_accessor = accessors[idx_accessor_idx]
        idx_buffer_view = buffer_views[idx_accessor['bufferView']]
        
        idx_offset = idx_buffer_view.get('byteOffset', 0) + idx_accessor.get('byteOffset', 0)
        idx_count = idx_accessor['count']
        
        # Determine component type
        component_type = idx_accessor['componentType']
        if component_type == 5121:  # UNSIGNED_BYTE
            dtype = np.uint8
            stride = 1
        elif component_type == 5123:  # UNSIGNED_SHORT
            dtype = np.uint16
            stride = 2
        elif component_type == 5125:  # UNSIGNED_INT
            dtype = np.uint32
            stride = 4
        else:
            raise ValueError(f"Unsupported index component type: {component_type}")
        
        indices = np.frombuffer(
            bin_data[idx_offset:idx_offset + idx_count * stride],
            dtype=dtype
        )
        
        # Reshape to triangles
        faces = indices.reshape(-1, 3)
        num_faces = len(faces)
    
    # Calculate bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    dimensions = max_coords - min_coords
    
    return {
        'vertices': vertices,
        'faces': faces,
        'num_vertices': len(vertices),
        'num_faces': num_faces,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'dimensions': dimensions,
        'center': (min_coords + max_coords) / 2
    }


def voxel_grid_to_glb(grid: np.ndarray,
                       threshold: float = 0.5,
                       output_path: Optional[str] = None) -> str:
    """
    Convert a voxel grid (from optimization result) to GLB format.
    
    Args:
        grid: 3D numpy array of density values.
        threshold: Density threshold for including voxels (default 0.5).
        output_path: Optional path for output file. If None, generates temp file.
    
    Returns:
        Path to the generated GLB file.
    """
    # Find voxels above threshold
    nz, ny, nx = grid.shape
    solid_voxels = np.argwhere(grid > threshold)
    
    if len(solid_voxels) == 0:
        raise ValueError("No voxels above threshold found in the grid.")
    
    # Generate vertices and faces for each solid voxel (as cubes)
    all_vertices = []
    all_indices = []
    vertex_offset = 0
    
    # Define cube vertices relative to origin
    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)
    
    # Define cube faces (triangles) with consistent winding
    cube_faces = [
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ]
    
    for voxel in solid_voxels:
        z, y, x = voxel
        
        # Translate cube vertices to voxel position
        verts = cube_vertices + np.array([x, y, z], dtype=np.float32)
        all_vertices.extend(verts.tolist())
        
        # Add faces with offset
        for face in cube_faces:
            all_indices.extend([f + vertex_offset for f in face])
        vertex_offset += 8
    
    # Convert to numpy arrays
    vertices = np.array(all_vertices, dtype=np.float32)
    indices = np.array(all_indices, dtype=np.uint32)
    
    # Calculate bounds for accessor
    min_pos = vertices.min(axis=0).tolist()
    max_pos = vertices.max(axis=0).tolist()
    
    # Create GLB structure
    # Binary buffer: vertices + indices
    vertices_bytes = vertices.tobytes()
    indices_bytes = indices.tobytes()
    
    # Align to 4-byte boundary
    vertices_padding = (4 - len(vertices_bytes) % 4) % 4
    indices_padding = (4 - len(indices_bytes) % 4) % 4
    
    bin_data = (
        vertices_bytes + b'\x00' * vertices_padding +
        indices_bytes + b'\x00' * indices_padding
    )
    
    # Create JSON structure
    gltf_json = {
        "asset": {
            "version": "2.0",
            "generator": "ToPy Topology Optimization"
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4  # TRIANGLES
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(indices),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(vertices_bytes) + vertices_padding,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(vertices_bytes) + vertices_padding,
                "byteLength": len(indices_bytes) + indices_padding,
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [{
            "byteLength": len(bin_data)
        }]
    }
    
    # Encode JSON
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    
    # Pad JSON to 4-byte boundary
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_padding
    
    # Create GLB file
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(),
                                    f"topy_result_{uuid.uuid4().hex[:8]}.glb")
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'glTF')  # magic
        f.write(struct.pack('<I', 2))  # version
        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
        f.write(struct.pack('<I', total_length))  # length
        
        # JSON chunk
        f.write(struct.pack('<I', len(json_bytes)))  # chunk length
        f.write(b'JSON')  # chunk type
        f.write(json_bytes)  # chunk data
        
        # BIN chunk
        f.write(struct.pack('<I', len(bin_data)))  # chunk length
        f.write(b'BIN\x00')  # chunk type
        f.write(bin_data)  # chunk data
    
    return output_path
