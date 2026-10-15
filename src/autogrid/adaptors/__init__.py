"""Adaptors between autogrid and other libraries, e.g. iris and cartopy."""

from .cartopy import cartopy_projection

__all__ = ["cartopy_projection"]
