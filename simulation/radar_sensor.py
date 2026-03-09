"""
Radar sensor physics and simulation

This module implements the radar sensor model including:
- SNR computation (radar equation)
- Detection probability (ROC curve)
- Beam footprint geometry
- Radar measurement simulation
"""

from __future__ import annotations
import numpy as np
from grid import get_grid_shape


def cartesian_to_polar(x, y):
    """
    Convert BEV Cartesian to polar coordinates
    
    COORDINATE SYSTEM (Standard Math Convention):
    - X: right (positive = right, negative = left)
    - Y: forward (positive = up/forward, negative = down/backward)
    - Origin: ego vehicle position
    
    Polar Convention:
    - r: Euclidean distance from ego vehicle
    - theta: Angle measured from positive Y-axis (forward direction)
      * 0° = straight ahead (positive Y)
      * +90° = right (positive X)
      * -90° = left (negative X)
      * ±180° = backward (negative Y)
    
    Args:
        x, y: Cartesian coordinates (meters) - can be arrays
              x = lateral (right/left), y = longitudinal (forward/back)
    
    Returns:
        r: Range (meters)
        theta: Azimuth angle (degrees) measured from forward direction
    """
    r = np.sqrt(x**2 + y**2)
    # arctan2(x, y) gives angle from Y-axis (not X-axis!)
    theta = np.degrees(np.arctan2(x, y))
    return r, theta


def polar_to_cartesian(r, theta_deg):
    """
    Convert polar to Cartesian coordinates
    
    Inverse of cartesian_to_polar().
    
    Polar Convention:
    - theta: Angle from forward direction (positive Y axis)
      * 0° = forward (positive Y)
      * +90° = right (positive X)
      * -90° = left (negative X)
    
    Args:
        r: Range (meters)
        theta_deg: Azimuth angle (degrees)
    
    Returns:
        x: Lateral coordinate (meters) - positive = right
        y: Longitudinal coordinate (meters) - positive = forward
    
    Math:
    Since theta is measured from Y-axis (not X-axis):
    - x = r * sin(theta)  [perpendicular to Y-axis]
    - y = r * cos(theta)  [parallel to Y-axis]
    """
    theta_rad = np.radians(theta_deg)
    x = r * np.sin(theta_rad)  # Lateral: right (+) / left (-)
    y = r * np.cos(theta_rad)  # Longitudinal: forward (+) / backward (-)
    return x, y


def compute_snr(distance, rcs, radar_config):
    """
    Compute Signal-to-Noise Ratio using simplified radar equation
    
    Physics:
    The radar equation relates received power to transmitted power:
    P_r = (P_t * G_t * G_r * λ² * RCS) / ((4π)³ * R⁴ * L)
    
    where:
    - P_t: Transmitted power
    - G_t, G_r: Transmit and receive antenna gains
    - λ: Wavelength
    - RCS: Radar cross section (target reflectivity)
    - R: Range to target
    - L: System losses
    
    Simplified Form:
    SNR = K * RCS / R⁴
    
    where K absorbs all constant factors (power, gains, wavelength, etc.)
    
    The R⁴ dependency is fundamental: signal travels to target (R²) and back (R²)
    
    Args:
        distance: Range to target (meters) - can be array
        rcs: Radar cross section (m²) - larger = more reflective
        radar_config: RadarConfig object with K_radar constant
    
    Returns:
        snr_db: Signal-to-noise ratio in decibels (dB)
    """
    # Clamp distance to avoid singularity at origin
    distance = np.maximum(distance, 1.0)  # Minimum 1 meter
    
    # Compute linear SNR
    snr_linear = (radar_config.K_radar * rcs) / (distance ** 4)
    
    # Convert to dB scale: SNR_dB = 10*log10(SNR_linear)
    snr_db = 10 * np.log10(snr_linear + 1e-10)  # Small epsilon for numerical stability
    
    return snr_db


def detection_probability(snr_db, threshold_db=-10):
    """
    Probability of detection given SNR
    
    Models the Receiver Operating Characteristic (ROC) curve using sigmoid.
    
    Theory:
    - Detection is a binary decision: signal present or not
    - Decision is made by comparing SNR to threshold
    - Real detectors have smooth transition (not hard threshold)
    - Sigmoid approximates this soft decision boundary
    
    Formula:
    P_d(SNR) = 1 / (1 + exp(-(SNR - threshold)))
    
    Behavior:
    - SNR >> threshold → P_d ≈ 1 (very likely to detect)
    - SNR = threshold → P_d = 0.5 (50/50 chance)
    - SNR << threshold → P_d ≈ 0 (unlikely to detect)
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        threshold_db: Detection threshold in dB (typically -10 to 10 dB)
    
    Returns:
        P_d: Probability of detection [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-(snr_db - threshold_db)))


def get_radar_footprint(beam_azimuth, beam_range, grid_conf, radar_config):
    """
    Calculate radar beam footprint in BEV grid
    
    This function defines the spatial region illuminated by a radar pulse.
    The footprint is determined by:
    1. Angular cone (beam width in azimuth)
    2. Range gate (depth window around target range)
    3. Field of view limits (maximum range)
    
    Physical Interpretation:
    - Radar transmits a conical beam (not a pencil beam!)
    - Beam width determined by antenna aperture size
    - Range resolution determined by pulse width/bandwidth
    - All cells within the cone receive radar energy
    
    Coordinate System:
    - X positive = right, X negative = left
    - Y positive = forward (up on plot), Y negative = backward
    - 0° azimuth = forward (positive Y direction)
    
    Args:
        beam_azimuth: Beam pointing direction (degrees)
                     0° = forward, +90° = right, -90° = left
        beam_range: Target range (meters) - center of range gate
        grid_conf: BEV grid configuration dict
        radar_config: RadarConfig object
    
    Returns:
        footprint_mask: (H, W) binary mask [0 or 1]
                       1 = cell is illuminated by this beam
        confidence_map: (H, W) detection confidence [0, 1]
                       Accounts for SNR-dependent detection probability
    """
    # BEV grid dimensions from config
    from grid import get_grid_shape
    H, W = get_grid_shape(grid_conf)
    resolution = grid_conf['xbound'][2]  # meters per pixel
    
    # Create coordinate grids for all BEV pixels
    x_coords = np.linspace(grid_conf['xbound'][0], grid_conf['xbound'][1], W)
    y_coords = np.linspace(grid_conf['ybound'][0], grid_conf['ybound'][1], H)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # NO FLIP NEEDED! Our coordinate system:
    # - X positive = right
    # - Y positive = forward (up on plot)
    # Meshgrid already gives correct orientation with origin='lower'
    
    # Convert all pixels to polar coordinates (range, azimuth)
    R, Theta = cartesian_to_polar(X, Y)
    
    # Beam geometry parameters
    beam_width = radar_config.azimuth_resolution  # degrees (e.g., 3°)
    range_gate = radar_config.range_gate_size  # meters - range window around target
    
    # Define beam cone (angular constraint)
    angular_diff = np.abs(Theta - beam_azimuth)
    # Handle angle wrapping (e.g., -170° and +170° are close!)
    angular_diff = np.minimum(angular_diff, 360 - angular_diff)
    in_cone = angular_diff <= beam_width / 2  # Within ±beam_width/2
    
    # Define range gate (radial constraint)
    in_range = (R >= beam_range - range_gate) & (R <= beam_range + range_gate)
    
    # Maximum range limit (hardware constraint)
    in_fov = R <= radar_config.max_range
    
    # Combine all constraints: footprint = cone ∩ range_gate ∩ fov
    footprint_mask = (in_range & in_cone & in_fov).astype(float)
    
    # Compute detection confidence based on SNR model
    # SNR depends on range (R^-4 law) and target reflectivity (RCS)
    snr = compute_snr(R, radar_config.rcs_vehicle, radar_config)
    confidence = detection_probability(snr, radar_config.snr_threshold_db)
    
    # Apply confidence only within footprint (outside footprint = no signal)
    confidence_map = footprint_mask * confidence
    
    return footprint_mask, confidence_map


def simulate_radar_return(beam_azimuth, beam_range, ground_truth, grid_conf, radar_config):
    """
    Simulate radar detection given ground truth occupancy
    
    This is the "forward model" that generates synthetic radar measurements.
    
    Physical Process:
    1. Radar illuminates a cone-shaped region (footprint)
    2. Objects in the footprint reflect energy back
    3. Detection confidence depends on:
       - Object presence (ground truth)
       - SNR (range-dependent)
       - Random noise and clutter
    
    Detection Cases:
    - True positive: Radar detects actual objects (high confidence)
    - False alarm: Radar detects in empty space (low confidence, P_fa = 5%)
    - Miss: Radar fails to detect objects (handled by SNR model)
    
    Args:
        beam_azimuth: Beam direction (degrees)
        beam_range: Target range (meters)
        ground_truth: (H, W) binary occupancy map [0, 1]
        grid_conf: BEV configuration
        radar_config: RadarConfig object
    
    Returns:
        detection_map: (H, W) detection confidence [0, 1]
                      0 = no detection or not scanned
                      >0 = detection strength
    """
    # Get beam footprint and SNR-based confidence
    footprint, confidence_map = get_radar_footprint(beam_azimuth, beam_range, 
                                                     grid_conf, radar_config)
    
    # Initialize detection map
    detection_map = np.zeros_like(ground_truth, dtype=float)
    illuminated = footprint > 0
    
    # CASE 1: True positives - radar detects actual objects
    true_pos = illuminated & (ground_truth > 0.5)
    detection_map[true_pos] = confidence_map[true_pos]
    
    # CASE 2: False alarms - clutter detections in empty space
    false_alarm = illuminated & (ground_truth <= 0.5)
    false_alarm_mask = np.random.rand(*ground_truth.shape) < radar_config.probability_of_false_alarm
    detection_map[false_alarm & false_alarm_mask] = 0.3  # Low confidence clutter
    
    # CASE 3: Add measurement noise ONLY to illuminated regions
    noise = np.random.normal(0, radar_config.inverse_model_noise_std, detection_map.shape)
    detection_map[illuminated] = np.clip(detection_map[illuminated] + noise[illuminated], 0, 1)
    
    return detection_map

