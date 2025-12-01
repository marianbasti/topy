#!/usr/bin/env python
"""
ToPy install script.

Install ToPy through `python setup.py install`.

For web application only (Python 3):
    pip install -e ".[webapp]"

For full ToPy with pysparse (Python 2.7):
    python setup.py install
"""

import json
import setuptools

# Get metadata.
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# Get project description.
with open("README.md", "r") as fh:
    long_description = fh.read()

# Base requirements (Python 3 compatible)
base_requires = [
    'numpy',
    'matplotlib',
    'sympy',
    'pyvtk',
]

# Web application requirements
webapp_requires = [
    'flask>=2.0.0',
    'scipy>=1.0.0',
    'plyfile>=0.7.0',
]

# Legacy requirements (Python 2.7 only)
legacy_requires = [
    'typing',
    'pathlib',
    'pysparse',
]

setuptools.setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=base_requires,
    extras_require={
        'webapp': webapp_requires,
        'legacy': legacy_requires,
        'all': webapp_requires + legacy_requires,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'topy-web=topy.webapp.run_server:main',
        ],
    },
    **metadata
)