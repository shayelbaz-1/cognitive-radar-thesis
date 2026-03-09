"""
Information-theoretic functions for active sensing

This module implements:
- Shannon entropy for uncertainty quantification
- Information gain calculation
- Bayesian sensor fusion
"""

import numpy as np


def compute_entropy(prob_map):
    """
    Compute binary entropy (Shannon entropy for binary variable)
    
    Formula:
    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)  [bits]
    
    Properties:
    - H(0) = H(1) = 0 bits (fully certain)
    - H(0.5) = 1 bit (maximum uncertainty)
    - Symmetric: H(p) = H(1-p)
    - Always non-negative
    
    Physical Interpretation:
    - Measures the expected information content of a binary outcome
    - High entropy = high uncertainty = need for sensing
    - Low entropy = low uncertainty = confident prediction
    
    Args:
        prob_map: (H, W) occupancy probability [0, 1]
                 p = probability that cell is occupied
    
    Returns:
        entropy: (H, W) entropy in bits [0, 1]
                Higher values = more uncertain
    """
    eps = 1e-7  # Small epsilon to avoid log(0)
    p = np.clip(prob_map, eps, 1 - eps)
    
    # Binary entropy formula
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    return entropy


def compute_information_gain(entropy_before, entropy_after):
    """
    Calculate information gain (entropy reduction)
    
    Formula:
    IG = H_before - H_after  [bits]
    
    Interpretation:
    - IG > 0: Uncertainty decreased (information was gained)
    - IG = 0: No learning occurred
    - IG < 0: Should not happen (entropy cannot increase without new uncertainty)
    
    This is the KEY METRIC for active sensing:
    - Cognitive radar aims to maximize IG per pulse
    - Higher IG = more efficient beam placement
    - Total IG quantifies overall learning effectiveness
    
    Args:
        entropy_before: (H, W) entropy before radar scan [bits]
        entropy_after: (H, W) entropy after fusion [bits]
    
    Returns:
        total_ig: Total information gain (scalar) [bits]
                 Sum over all pixels - measures global learning
        spatial_ig: (H, W) per-pixel information gain [bits]
                   Shows WHERE learning occurred
    """
    # Compute spatial information gain
    spatial_ig = entropy_before - entropy_after
    
    # Ensure non-negative (handle numerical errors)
    # In theory, entropy should never increase, but rounding errors can cause small negatives
    spatial_ig = np.maximum(spatial_ig, 0)
    
    # Total information gain (sum over all pixels)
    total_ig = spatial_ig.sum()
    
    return total_ig, spatial_ig


def bayesian_fusion_raytracing(camera_prob, radar_measurement, confidence=0.85):
    """
    Physically-correct Bayesian fusion using inverse sensor model
    
    PHYSICS-BASED UPDATE:
    Radar provides THREE types of information per pulse:
    
    1. FREE SPACE (along ray before hit):
       - Evidence of EMPTY space
       - Decrease occupancy probability
       - LR < 1 (negative evidence)
    
    2. HIT (obstacle detection):
       - Evidence of OCCUPIED space
       - Increase occupancy probability
       - LR > 1 (positive evidence)
    
    3. SHADOW (behind obstacle):
       - NO INFORMATION (line-of-sight blocked)
       - DO NOT UPDATE (keep prior)
       - LR = 1
    
    This is the CORRECT inverse sensor model for radar!
    
    THEORY:
    Bayes' Rule (odds form):
    posterior_odds = prior_odds × likelihood_ratio
    
    Args:
        camera_prob: (H, W) prior occupancy probability [0, 1]
        radar_measurement: Dict from radar_inverse_sensor_model() with keys:
            - 'free_space': (H, W) binary mask of empty cells along ray
            - 'occupied': (H, W) detection confidence at hit
            - 'shadow': (H, W) binary mask of occluded cells
        confidence: Radar reliability [0, 1]
    
    Returns:
        fused_prob: (H, W) posterior occupancy probability [0, 1]
    """
    eps = 1e-7
    
    # Convert to odds
    camera_odds = camera_prob / (1 - camera_prob + eps)
    
    # Initialize likelihood ratio to 1 (no update by default)
    likelihood_ratio = np.ones_like(camera_prob)
    
    # CASE 1: FREE SPACE → Strong evidence for EMPTY
    # Entire beam cone traveled through without hitting
    free_space = radar_measurement['free_space']
    likelihood_ratio[free_space] = 0.01  # LR << 1 (very strong negative evidence)
    
    # CASE 2: OCCUPANCY SPLAT (Thick Arc)
    # The splat is already built into the measurement by radar_inverse_sensor_model
    # It represents the entire beam width × car depth where detection occurred
    strong_hit = radar_measurement['occupied'] > 0.7
    medium_hit = (radar_measurement['occupied'] > 0.3) & (radar_measurement['occupied'] <= 0.7)
    weak_hit = (radar_measurement['occupied'] > 0.1) & (radar_measurement['occupied'] <= 0.3)
    
    # Apply likelihood ratios based on detection confidence
    # Balance: Radar is a DIRECT measurement, but camera has more context
    # Bayesian fusion should combine both sources intelligently
    likelihood_ratio[strong_hit] = 100.0   # Very strong evidence (thick arc from radar)
    likelihood_ratio[medium_hit] = 20.0    # Medium evidence
    likelihood_ratio[weak_hit] = 5.0       # Weak evidence (clutter/false alarms)
    
    # CASE 3: SHADOW → NO UPDATE
    # Behind obstacle, no line-of-sight → keep prior
    shadow = radar_measurement['shadow']
    likelihood_ratio[shadow] = 1.0  # No information
    
    # SAFEGUARD: Ensure we only update cells that radar actually measured
    # (free space, hit, or shadow). Everything else keeps prior (LR=1)
    measured_region = free_space | (radar_measurement['occupied'] > 0) | shadow
    unmeasured = ~measured_region
    likelihood_ratio[unmeasured] = 1.0  # No change to prior
    
    # Apply Bayesian update
    updated_odds = camera_odds * likelihood_ratio
    fused_prob = updated_odds / (1 + updated_odds)
    
    return fused_prob

