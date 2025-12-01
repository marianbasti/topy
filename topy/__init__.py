"""
# ==============================================================================
# ToPy -- Topology optimization with Python.
# Copyright (C) 2012, 2015, 2016, 2017 William Hunter.
# ==============================================================================
"""

# Core ToPy modules are imported conditionally since they require pysparse
# which may not be available in Python 3
try:
    from .topology import *
    from .visualisation import *
    from .elements import *
    from .optimisation import *
    
    __all__ = (
        topology.__all__ +
        visualisation.__all__ +
        elements.__all__ +
        optimisation.__all__
    )
except ImportError:
    # Core modules not available (likely missing pysparse)
    # Webapp can still function with its own optimization implementation
    __all__ = []

__version__ = "0.5.0"
__author__  = "William Hunter <whunter.za at gmail dot com>"
