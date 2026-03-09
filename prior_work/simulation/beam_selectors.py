"""
Beam selection strategies for radar simulation

This module implements baseline (open-loop) beam selection strategies:
- UniformSelector: Fixed angular scan pattern with uniform spacing
- RandomSelector: Random beam selection within FOV

Physical model:
- Beams are specified by AZIMUTH ANGLE only (degrees from forward)
- Radar rays originate from ego vehicle and propagate outward
- Ray tracing determines range-to-hit, free space, and shadows

Note: The cognitive (entropy-guided) strategy uses adaptive single-beam
selection in the main simulation loop, not these pre-planned selectors.
"""

import numpy as np


class UniformSelector:
    """
    Uniform scanning baseline - fixed angular pattern
    
    Strategy:
    - Pre-plans all beams at once (open-loop)
    - Uniform angular spacing across FOV
    - Does NOT adapt to entropy or measurements
    
    PHYSICS: Each beam is just an AZIMUTH ANGLE (ray direction)
    """
    
    def __init__(self, radar_config, budget=10):
        """
        Args:
            radar_config: RadarConfig object
            budget: Number of beams to plan
        """
        self.radar_config = radar_config
        self.budget = budget
    
    def select_beams(self, entropy_map, grid_conf):
        """
        Generate uniform angular scan pattern
        
        Args:
            entropy_map: (H, W) - not used, for interface consistency
            grid_conf: BEV configuration
        
        Returns:
            beams: List of azimuth angles (degrees)
        """
        # Uniform angular spacing across FOV
        azimuth_max = self.radar_config.azimuth_fov / 2
        azimuths = np.linspace(-azimuth_max, azimuth_max, self.budget)
        
        return list(azimuths)


class RandomSelector:
    """
    Random scanning baseline - lower bound performance
    
    Strategy:
    - Pre-plans all beams at once (open-loop)
    - Random azimuth angles within FOV
    - Does NOT adapt to entropy or measurements
    
    PHYSICS: Each beam is just an AZIMUTH ANGLE (ray direction)
    """
    
    def __init__(self, radar_config, budget=10):
        """
        Args:
            radar_config: RadarConfig object
            budget: Number of beams to plan
        """
        self.radar_config = radar_config
        self.budget = budget
    
    def select_beams(self, entropy_map, grid_conf):
        """
        Generate random beam pattern
        
        Args:
            entropy_map: (H, W) - not used, for interface consistency
            grid_conf: BEV configuration
        
        Returns:
            beams: List of azimuth angles (degrees)
        """
        # Random azimuth angles within FOV
        azimuth_max = self.radar_config.azimuth_fov / 2
        azimuths = np.random.uniform(-azimuth_max, azimuth_max, self.budget)
        
        return list(azimuths)

