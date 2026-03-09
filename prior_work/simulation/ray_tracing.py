"""
Ray tracing for physically-correct radar simulation

This module implements:
1. Ray casting from ego vehicle along specified azimuth
2. Line-of-sight occlusion checking
3. Free space identification (empty along ray)
4. Shadow region identification (occluded behind obstacles)

PHYSICS:
- Radar is a WAVE that travels from origin outward
- Cannot "skip" intermediate space to reach far regions
- Cannot see through obstacles (line-of-sight required)
- Returns are from FIRST object hit (everything behind is shadowed)
"""

from __future__ import annotations
import numpy as np
from grid import get_grid_shape, create_meshgrid


def bresenham_ray(x0, y0, x1, y1):
    """
    Bresenham's line algorithm for ray tracing
    
    Generates all grid cells along a line from (x0, y0) to (x1, y1).
    This is used for ray casting in discrete grid.
    
    Args:
        x0, y0: Start point (grid indices)
        x1, y1: End point (grid indices)
    
    Returns:
        cells: List of (x, y) tuples along the ray
    """
    cells = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    x, y = x0, y0
    
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    
    if dx > dy:
        error = dx / 2
        while x != x1:
            cells.append((x, y))
            error -= dy
            if error < 0:
                y += y_inc
                error += dx
            x += x_inc
    else:
        error = dy / 2
        while y != y1:
            cells.append((x, y))
            error -= dx
            if error < 0:
                x += x_inc
                error += dy
            y += y_inc
    
    cells.append((x, y))
    return cells


def cast_radar_cone(azimuth_deg, ground_truth, grid_conf, radar_config):
    """
    Cast a radar CONE (not a single ray!) along specified azimuth with proper physics
    
    CRITICAL FIX: "Cone Integration" to prevent "Threading the Needle" bug
    
    At long ranges (100m), a 3° beam becomes ~5m wide. A single centerline ray
    can miss cars that are inside the cone but offset from center. This function
    checks the ENTIRE CONE VOLUME to find the closest obstacle.
    
    PHYSICS MODEL:
    1. Beam is a CONE (angular width = beam_width)
    2. Check ALL pixels within the cone
    3. Find MINIMUM range to any occupied pixel
    4. Everything before hit is FREE SPACE (empty)
    5. Everything behind hit is SHADOWED (unknown)
    
    This is PHYSICALLY CORRECT: Radar has precise range but blurry angle.
    
    Args:
        azimuth_deg: Beam centerline direction in degrees (0 = forward, +90 = right)
        ground_truth: (H, W) binary occupancy map [0, 1]
        grid_conf: BEV grid configuration
        radar_config: Radar configuration with beam_width
    
    Returns:
        ray_result: Dict with keys:
            - 'free_space': (H, W) binary mask of free space in cone before hit
            - 'hit': (H, W) binary mask of hit region
            - 'shadow': (H, W) binary mask of shadowed region behind hit
            - 'hit_range': Distance to hit in meters (None if no hit)
            - 'ray_cells': List of (i, j) cells along CENTERLINE (for visualization)
    """
    H, W = ground_truth.shape
    resolution = grid_conf['xbound'][2]
    max_range = radar_config.max_range
    beam_width_deg = radar_config.azimuth_resolution  # e.g., 3°
    occupancy_threshold = radar_config.occupancy_threshold
    
    # Grid center is at (H//2, W//2) = ego vehicle position
    i0, j0 = H // 2, W // 2
    
    # ===================================================================
    # STEP 1: CREATE GEOMETRIC CONE MASK
    # ===================================================================
    # Instead of a single ray, create the ENTIRE CONE VOLUME
    
    # Create coordinate grids
    x_coords = np.linspace(grid_conf['xbound'][0], grid_conf['xbound'][1], W)
    y_coords = np.linspace(grid_conf['ybound'][0], grid_conf['ybound'][1], H)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Convert to polar coordinates (R, Theta) from ego position
    X_ego = X - 0  # X relative to ego (ego is at world origin 0,0)
    Y_ego = Y - 0  # Y relative to ego
    R = np.sqrt(X_ego**2 + Y_ego**2)
    Theta = np.degrees(np.arctan2(X_ego, Y_ego))  # Angle in degrees
    
    # Define cone: all pixels within angular bounds
    # Beam cone extends from (azimuth - beam_width/2) to (azimuth + beam_width/2)
    half_width = beam_width_deg / 2.0
    angle_min = azimuth_deg - half_width
    angle_max = azimuth_deg + half_width
    
    # Handle angle wrapping (e.g., -180° to +180°)
    angular_diff = Theta - azimuth_deg
    angular_diff = (angular_diff + 180) % 360 - 180  # Wrap to [-180, 180]
    
    # Cone mask: pixels within angular bounds and range bounds
    cone_mask = (np.abs(angular_diff) <= half_width) & \
                (R >= radar_config.min_range) & \
                (R <= max_range)
    
    # ===================================================================
    # STEP 2: FIND MINIMUM RANGE TO ANY OCCUPIED PIXEL IN CONE
    # ===================================================================
    # Check ALL pixels in the cone, find the closest obstacle
    
    occupied_in_cone = cone_mask & (ground_truth > occupancy_threshold)
    
    if occupied_in_cone.any():
        # HIT! Find the minimum range to any occupied pixel
        occupied_ranges = R[occupied_in_cone]
        hit_range = occupied_ranges.min()
    else:
        # NO HIT - entire cone is clear
        hit_range = None
    
    # ===================================================================
    # STEP 3: CREATE FREE SPACE, HIT, AND SHADOW MASKS
    # ===================================================================
    
    free_space = np.zeros((H, W), dtype=bool)
    hit = np.zeros((H, W), dtype=bool)
    shadow = np.zeros((H, W), dtype=bool)
    
    if hit_range is not None:
        # FREE SPACE: Entire cone from min_range to hit_range
        free_space = cone_mask & (R < hit_range)
        
        # HIT: Thick arc at hit_range (assumed car depth from config)
        hit_tolerance = radar_config.assumed_car_depth  # meters
        hit = cone_mask & (R >= hit_range) & (R < hit_range + hit_tolerance)
        
        # SHADOW: Entire cone beyond hit
        shadow = cone_mask & (R >= hit_range + hit_tolerance)
    else:
        # No hit: entire cone is free space
        free_space = cone_mask.copy()
    
    # ===================================================================
    # STEP 4: GENERATE CENTERLINE RAY FOR VISUALIZATION
    # ===================================================================
    # Still use Bresenham for drawing the beam centerline in plots
    
    theta_rad = np.radians(azimuth_deg)
    x_end = max_range * np.sin(theta_rad)
    y_end = max_range * np.cos(theta_rad)
    
    j_end = int(j0 + x_end / resolution)
    i_end = int(i0 + y_end / resolution)
    j_end = np.clip(j_end, 0, W - 1)
    i_end = np.clip(i_end, 0, H - 1)
    
    ray_cells = bresenham_ray(j0, i0, j_end, i_end)
    
    return {
        'free_space': free_space,
        'hit': hit,
        'shadow': shadow,
        'hit_range': hit_range,
        'ray_cells': ray_cells,
        'azimuth': azimuth_deg
    }


def compute_visibility_mask(belief_map, grid_conf, radar_config, sim_config=None):
    """
    Compute P_visible for each cell based on CURRENT BELIEF (not GT!)
    
    Physics: P_visible(r,θ) = probability a ray reaches cell (r,θ)
    
    Algorithm:
    1. Cast rays in all directions through current belief map
    2. Cells along ray BEFORE high-probability obstacles = visible
    3. Cells BEHIND likely obstacles = invisible (expected shadow)
    
    This prevents targeting occluded high-entropy regions
    ("fly-at-the-window" problem).
    
    Args:
        belief_map: (H, W) current occupancy belief [0, 1]
        grid_conf: BEV grid configuration
        radar_config: Radar configuration
        sim_config: Simulation configuration (optional, for num_rays parameter)
    
    Returns:
        visibility_mask: (H, W) float mask [0, 1]
                        1.0 = fully visible
                        0.0 = likely occluded
    """
    H, W = belief_map.shape
    visibility_mask = np.zeros((H, W), dtype=bool)
    
    # Cast rays at regular intervals to compute visibility
    num_rays = sim_config.visibility_num_rays_belief if sim_config else 180
    azimuths = np.linspace(-180, 180, num_rays, endpoint=False)
    
    for azimuth in azimuths:
        # Ray cast through BELIEF (not GT!) to predict occlusion
        ray_result = cast_radar_cone(azimuth, belief_map, grid_conf, radar_config)
        
        # Visible: free space + first obstacle (can potentially measure)
        visibility_mask |= ray_result['free_space']
        visibility_mask |= ray_result['hit']
        # Shadow: NOT visible (occluded by expected obstacle)
    
    return visibility_mask.astype(float)


def compute_gt_visibility_mask(ground_truth, grid_conf, radar_config, sim_config=None):
    """
    Compute what is physically REACHABLE from ego using GROUND TRUTH
    
    This defines the "THEORETICAL MAXIMUM" - what we COULD see if we:
    - Had unlimited radar pulses
    - Made perfect beam selections
    - Had perfect sensor accuracy
    
    The only limit is PHYSICS: we cannot see through walls/cars.
    
    This is used to compute the "Glass Ceiling" - the best possible
    performance given occlusions in the scene.
    
    Args:
        ground_truth: (H, W) true occupancy [0, 1]
        grid_conf: BEV grid configuration
        radar_config: Radar configuration
        sim_config: Simulation configuration (optional, for num_rays parameter)
    
    Returns:
        visibility_mask: (H, W) boolean mask
                        True = physically visible from ego
                        False = occluded by obstacles
    """
    H, W = ground_truth.shape
    visibility_mask = np.zeros((H, W), dtype=bool)
    
    # Cast rays in all directions through GROUND TRUTH
    num_rays = sim_config.visibility_num_rays_gt if sim_config else 360
    azimuths = np.linspace(-180, 180, num_rays, endpoint=False)
    
    for azimuth in azimuths:
        # Ray cast through GROUND TRUTH to find occlusions
        ray_result = cast_radar_cone(azimuth, ground_truth, grid_conf, radar_config)
        
        # Visible: free space + first obstacle hit
        # (Everything up to and including the first obstacle is visible)
        visibility_mask |= ray_result['free_space']
        visibility_mask |= ray_result['hit']
        # Shadow regions are NOT included (occluded)
    
    return visibility_mask


def radar_inverse_sensor_model(azimuth_deg, ground_truth, grid_conf, radar_config,
                                detection_confidence=None, false_alarm_rate=None):
    """
    Simulate physically-correct radar return
    
    PHYSICS - THE INDIVISIBLE BEAM:
    A radar pulse is ONE measurement, not multiple rays!
    The radar integrates energy from the ENTIRE beam cone.
    
    1. Use cast_radar_cone to find IF there's a hit and WHERE (range)
    2. IF HIT: Create "THICK ARC" splat (entire beam width × car depth)
    3. IF NO HIT: Clear entire cone as free space
    
    This matches real radar physics: you cannot distinguish returns
    from different parts of the beam width.
    
    OPTIMIZATION: Trusts cast_radar_cone's geometry calculations instead of
    recalculating R, Theta, and cone masks (eliminates redundant computation).
    
    Args:
        azimuth_deg: Beam centerline direction (degrees)
        ground_truth: (H, W) true occupancy
        grid_conf: BEV configuration
        radar_config: Radar configuration
        detection_confidence: P(detect | occupied) for hit
        false_alarm_rate: P(detect | empty) for false alarms
    
    Returns:
        measurement: Dict with free_space, occupied, shadow masks
    """
    H, W = ground_truth.shape
    
    # Use config defaults if not provided
    if detection_confidence is None:
        detection_confidence = radar_config.probability_of_detection
    if false_alarm_rate is None:
        false_alarm_rate = radar_config.false_alarm_rate
    
    # Cast cone - this does ALL the geometry calculations (no need to repeat!)
    cone_result = cast_radar_cone(azimuth_deg, ground_truth, grid_conf, radar_config)
    
    # Initialize occupancy measurement (starts at zero everywhere)
    occupied_measurement = np.zeros((H, W), dtype=float)
    
    if cone_result['hit_range'] is not None:
        # === CASE 1: HIT DETECTED ===
        # Use the hit mask from cone_result directly
        occupancy_splat = cone_result['hit']
        
        # Fill with detection confidence
        occupied_measurement[occupancy_splat] = detection_confidence
        
        # Add measurement noise to occupancy
        noise_std = radar_config.inverse_model_noise_std
        noise = np.random.normal(0, noise_std, size=(H, W))
        occupied_measurement[occupancy_splat] = np.clip(
            occupied_measurement[occupancy_splat] + noise[occupancy_splat], 0.7, 1.0
        )
        
    else:
        # === CASE 2: NO HIT (Silence) ===
        # Rare false alarms in free space
        false_alarms = cone_result['free_space'] & (np.random.rand(H, W) < false_alarm_rate)
        occupied_measurement[false_alarms] = 0.3  # Low confidence clutter
    
    return {
        'free_space': cone_result['free_space'],
        'occupied': occupied_measurement,
        'shadow': cone_result['shadow'],
        'azimuth': azimuth_deg,
        'hit_range': cone_result['hit_range'],
        'ray_cells': cone_result['ray_cells']  # For visualization
    }

