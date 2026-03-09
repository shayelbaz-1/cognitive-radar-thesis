"""
Grid utilities for BEV (Bird's-Eye View) coordinate systems

This module provides helper functions for working with BEV grids:
- Computing grid dimensions from configuration
- Creating coordinate meshgrids
- Converting between grid and world coordinates
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def get_grid_shape(grid_conf: dict) -> Tuple[int, int]:
    """
    Compute grid dimensions (H, W) from grid configuration
    
    Args:
        grid_conf: Grid configuration dict with 'xbound', 'ybound' keys
                  Each bound is [min, max, resolution]
    
    Returns:
        (H, W): Grid height and width in pixels
    
    Example:
        >>> grid_conf = {'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5]}
        >>> get_grid_shape(grid_conf)
        (200, 200)
    """
    xbound = grid_conf['xbound']
    ybound = grid_conf['ybound']
    
    # W = number of x bins, H = number of y bins
    W = int((xbound[1] - xbound[0]) / xbound[2])
    H = int((ybound[1] - ybound[0]) / ybound[2])
    
    return H, W


def create_meshgrid(grid_conf: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create coordinate meshgrids for BEV grid
    
    Args:
        grid_conf: Grid configuration dict with 'xbound', 'ybound' keys
    
    Returns:
        (X, Y): Meshgrids of world coordinates
                X[i, j] = x-coordinate of pixel (i, j)
                Y[i, j] = y-coordinate of pixel (i, j)
    
    Convention:
        - X: lateral (right/left), positive = right
        - Y: longitudinal (forward/back), positive = forward
        - Origin: ego vehicle position
    """
    H, W = get_grid_shape(grid_conf)
    
    x_coords = np.linspace(grid_conf['xbound'][0], grid_conf['xbound'][1], W)
    y_coords = np.linspace(grid_conf['ybound'][0], grid_conf['ybound'][1], H)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    
    return X, Y


def get_resolution(grid_conf: dict) -> float:
    """
    Get grid resolution in meters per pixel
    
    Args:
        grid_conf: Grid configuration dict
    
    Returns:
        resolution: Meters per pixel (assumes square pixels)
    """
    return grid_conf['xbound'][2]
