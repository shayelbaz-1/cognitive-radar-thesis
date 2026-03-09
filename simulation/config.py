"""
Configuration classes for radar simulation

This module defines all simulation parameters and radar specifications.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RadarConfig:
    """
    Radar sensor parameters (based on 77GHz automotive radar)
    
    These values are chosen to match typical automotive FMCW radar specifications:
    - Operating frequency: 77 GHz (W-band)
    - Range: 4-50m (short to mid-range)
    - Angular resolution: 3° (typical for commercial systems)
    - FOV: 120° azimuth (wide coverage for driving scenarios)
    
    SNR Model:
    - Simplified radar equation: SNR = K * RCS / R^4
    - K_radar includes transmit power, antenna gains, wavelength, losses
    - Detection threshold: -10 dB (standard for automotive radar)
    """
    
    # Range parameters
    max_range: float = 100.0  # meters - maximum detection range
    range_resolution: float = 1.0  # meters - range bin size
    min_range: float = 1.0  # meters - minimum detection range (near-field limit)
    
    # Angular parameters
    azimuth_fov: int = 120  # degrees - total horizontal field of view (-60° to +60°)
    azimuth_resolution: int = 3  # degrees - beam width (angular resolution)
    elevation_fov: int = 20  # degrees - vertical field of view (not used in 2D BEV)
    
    # Detection parameters
    probability_of_detection: float = 0.9  # P_d for objects with good RCS (high SNR)
    probability_of_false_alarm: float = 0.05  # P_fa for clutter/noise
    
    # Radar Cross Section (RCS) - typical values in m²
    # RCS determines how much power is reflected back to the radar
    rcs_vehicle: float = 10.0  # cars/trucks (large, metallic)
    rcs_pedestrian: float = 0.5  # people (small, non-metallic)
    rcs_background: float = 0.01  # clutter/noise floor
    
    # SNR model parameters
    K_radar: float = 1e10  # Radar equation constant (absorbs Pt, Gt, Gr, λ, losses)
    snr_threshold_db: float = -10  # Detection threshold in dB (Neyman-Pearson criterion)
    
    # Thick Arc Parameters (for physically-correct beam measurement)
    assumed_car_depth: float = 4.0  # meters - assumed object depth for thick arc splat
    
    # Segmentation threshold
    occupancy_threshold: float = 0.5  # Threshold for binary segmentation (0-1 probability)
    
    # Sensor/noise parameters (for inverse sensor model and detection simulation)
    inverse_model_noise_std: float = 0.05  # Std dev for measurement noise
    false_alarm_rate: float = 0.05  # Rate of false alarms in free space
    range_gate_size: float = 5.0  # meters - range window around target (±5m)


@dataclass
class SimulationConfig:
    """
    Simulation parameters for radar experiment
    
    Beam Budget:
    - Constrains how many radar pulses can be used per scene
    - Simulates real-world limitations (energy, time, computational cost)
    - Cognitive strategy must be smart about where to look!
    
    BEV Grid:
    - Bird's-Eye View coordinate system (NuScenes convention)
    - X: forward (positive = ahead of vehicle)
    - Y: left (positive = left side, negative = right side)
    - Resolution: 0.5m (200x200 grid covering 100m x 100m)
    
    Radar Confidence:
    - Weighs radar measurements in Bayesian fusion
    - 0.85 = radar is trusted but not perfect (allows camera to contribute)
    - Higher = trust radar more, Lower = trust camera more
    """
    
    # Beam budget (resource constraint)
    radar_budget: int = 10  # Number of beams per scene (pulses allowed)
    
    # BEV grid configuration (NuScenes format) - using field(default_factory=...) for mutable default
    grid_conf: dict = None
    
    # Bayesian fusion parameter
    radar_confidence: float = 0.85  # Trustworthiness weight for radar measurements [0, 1]
    
    # Output configuration
    results_dir: str = None  # Will be set by main() based on experiment name and timestamp
    num_test_scenes: int = 50  # Number of validation scenes to test
    num_examples_to_save: int = 5  # Number of example visualizations per strategy
    
    # Thresholds for metrics and analysis
    segmentation_threshold: float = 0.5  # Threshold for binarizing predictions
    high_entropy_threshold: float = 0.5  # Threshold to classify high entropy regions
    
    # Visibility computation parameters (ray casting for occlusion)
    visibility_num_rays_belief: int = 180  # Rays for visibility mask from belief (2° resolution)
    visibility_num_rays_gt: int = 360  # Rays for GT visibility mask (1° resolution)
    
    # Beam selection parameters (for cognitive strategy)
    candidate_azimuth_count: int = 60  # Number of candidate azimuths to evaluate (2° resolution)
    revisit_angle_deg: float = 3.0  # Angular threshold for "revisit" penalty
    revisit_penalty: float = 0.3  # Penalty factor for revisiting scanned angles
    target_utility_scale: float = 5.0  # Scale factor for target-based utility weighting
    
    # Visualization control
    visualization_level: Literal["none", "final_only", "debug"] = "final_only"
    
    def __post_init__(self):
        """Initialize mutable defaults after dataclass initialization"""
        if self.grid_conf is None:
            self.grid_conf = {
                'xbound': [-50, 50, 0.5],  # [min, max, resolution] in meters (forward/back)
                'ybound': [-50, 50, 0.5],  # [min, max, resolution] in meters (left/right)
                'zbound': [-10, 10, 20],   # Height bounds (not used in 2D BEV)
                'dbound': [4.0, 45.0, 1.0] # Depth bounds for lifting (camera to BEV projection)
            }

