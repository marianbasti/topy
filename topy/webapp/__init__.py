"""
ToPy Web Application.

A Flask-based web interface for the ToPy topology optimization framework.
Allows users to upload 3D mesh files (PLY), configure optimization parameters,
run optimization, and download results.
"""

from .app import create_app

__all__ = ['create_app']
