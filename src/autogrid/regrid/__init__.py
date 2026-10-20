"""Regridding between Grids.

- regridder: Regridder, for converting arrays from one Grid to another.
- methods: the regridding methods a Regridder can apply.
"""

from autogrid.regrid.methods import Method
from autogrid.regrid.regridder import Regridder

__all__ = [
    "Method",
    "Regridder",
]
