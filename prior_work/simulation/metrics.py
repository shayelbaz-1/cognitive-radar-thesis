"""
Metric computation functions for radar simulation evaluation

This module provides all metric computation functions for evaluating
perception performance including:
- Industry standard metrics (F1, Precision, Recall, IoU)
- Target-focused metrics (ROI entropy, target-only error)
- Geometric metrics (Chamfer distance)
- Glass ceiling metrics (visible-only performance)
- ROC curves
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from config import RadarConfig


def compute_segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute F1-Score, Precision, Recall for binary segmentation
    
    Args:
        prediction: (H, W) predicted occupancy probability [0, 1]
        ground_truth: (H, W) true occupancy [0, 1]
        threshold: Threshold for binarization (default 0.5)
    
    Returns:
        dict with precision, recall, f1_score, TP, FP, FN
    """
    pred_binary = (prediction > threshold).astype(float)
    gt_binary = (ground_truth > threshold).astype(float)
    
    # True Positives: Predicted occupied AND actually occupied
    TP = (pred_binary * gt_binary).sum()
    
    # False Positives: Predicted occupied BUT actually empty
    FP = (pred_binary * (1 - gt_binary)).sum()
    
    # False Negatives: Predicted empty BUT actually occupied
    FN = ((1 - pred_binary) * gt_binary).sum()
    
    # Precision: Of all predicted occupied, how many are correct?
    precision = TP / (TP + FP + 1e-7)
    
    # Recall: Of all true occupied, how many did we find?
    recall = TP / (TP + FN + 1e-7)
    
    # F1-Score: Harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'TP': float(TP),
        'FP': float(FP),
        'FN': float(FN)
    }


def compute_target_only_error(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute MAE only on occupied cells (the targets)
    
    This fixes the "clown metric" problem where 95% empty road
    gives artificially good scores.
    
    Args:
        prediction: (H, W) predicted occupancy probability
        ground_truth: (H, W) true occupancy
        threshold: Threshold for identifying targets
    
    Returns:
        MAE only on occupied cells
    """
    gt_binary = (ground_truth > threshold)
    
    if gt_binary.sum() == 0:
        return 0.0  # No targets in scene
    
    # Compute error only where GT is occupied
    error_on_targets = np.abs(prediction[gt_binary] - ground_truth[gt_binary])
    return float(error_on_targets.mean())


def compute_roi_entropy(
    entropy_map: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute entropy only inside target bounding boxes
    
    Proves the agent is "resolving targets" not just "gathering data"
    
    Args:
        entropy_map: (H, W) Shannon entropy per cell
        ground_truth: (H, W) true occupancy
        threshold: Threshold for identifying targets
    
    Returns:
        Mean entropy only in occupied regions
    """
    gt_binary = (ground_truth > threshold)
    
    if gt_binary.sum() == 0:
        return 0.0  # No targets in scene
    
    # Entropy only where targets exist
    roi_entropy = entropy_map[gt_binary]
    return float(roi_entropy.mean())


def compute_chamfer_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Chamfer Distance - average distance from predicted to GT occupied pixels
    
    Measures geometric closeness. Less harsh than IoU for small offsets.
    
    Args:
        prediction: (H, W) predicted occupancy probability
        ground_truth: (H, W) true occupancy
        threshold: Threshold for binarization
    
    Returns:
        Average L2 distance (in pixels)
    """
    pred_binary = (prediction > threshold)
    gt_binary = (ground_truth > threshold)
    
    # Get coordinates of occupied pixels
    pred_coords = np.argwhere(pred_binary)
    gt_coords = np.argwhere(gt_binary)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0  # Can't compute distance if one set is empty
    
    # Chamfer Distance: For each predicted point, find nearest GT point
    from scipy.spatial.distance import cdist
    distances = cdist(pred_coords, gt_coords, metric='euclidean')
    
    # Average minimum distance from prediction to GT
    chamfer_pred_to_gt = distances.min(axis=1).mean()
    
    # Average minimum distance from GT to prediction (symmetric)
    chamfer_gt_to_pred = distances.min(axis=0).mean()
    
    # Chamfer distance is the average of both directions
    chamfer = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2.0
    
    return float(chamfer)


def compute_roc_curve(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_thresholds: int = 50
) -> Dict[str, np.ndarray | float]:
    """
    Compute ROC curve (TPR vs FPR) at multiple thresholds
    
    Args:
        prediction: (H, W) predicted occupancy probability
        ground_truth: (H, W) true occupancy
        num_thresholds: Number of threshold points
    
    Returns:
        dict with fpr, tpr, thresholds, auc
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_list = []  # True Positive Rate (Recall)
    fpr_list = []  # False Positive Rate
    
    gt_binary = (ground_truth > 0.5)
    
    for threshold in thresholds:
        pred_binary = (prediction > threshold)
        
        # True Positives / False Positives / False Negatives / True Negatives
        TP = (pred_binary & gt_binary).sum()
        FP = (pred_binary & (~gt_binary)).sum()
        FN = ((~pred_binary) & gt_binary).sum()
        TN = ((~pred_binary) & (~gt_binary)).sum()
        
        # TPR = TP / (TP + FN) = Recall
        tpr = TP / (TP + FN + 1e-7)
        
        # FPR = FP / (FP + TN)
        fpr = FP / (FP + TN + 1e-7)
        
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr_list, fpr_list)
    
    return {
        'fpr': np.array(fpr_list),
        'tpr': np.array(tpr_list),
        'thresholds': thresholds,
        'auc': float(abs(auc))  # abs because integration might be negative
    }


def compute_perfect_sensor_baseline(
    ground_truth: np.ndarray,
    gt_visibility_mask: np.ndarray,
    radar_config: RadarConfig,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute the TRUE THEORETICAL MAXIMUM given sensor limitations
    
    This simulates what we'd get with:
    - PERFECT beam placement (dense uniform coverage)
    - IMPERFECT sensor (Pd < 1.0, Pfa > 0, finite resolution)
    
    The "glass ceiling" is NOT 1.0 because:
    1. Radar detection probability < 1.0 (some targets missed)
    2. False alarms > 0 (some false positives)
    3. 3° beam width (spatial blurring)
    4. Thick arc approximation (not pixel-perfect)
    
    This defines the ACHIEVABLE maximum, accounting for sensor physics.
    
    Args:
        ground_truth: (H, W) true occupancy
        gt_visibility_mask: (H, W) visible AND in FOV mask
        radar_config: Radar configuration with Pd, Pfa
        threshold: Binarization threshold
    
    Returns:
        dict with theoretical maximum F1, IoU, Pd, Pfa
    """
    H, W = ground_truth.shape
    
    # Extract visible region
    gt_visible = ground_truth[gt_visibility_mask]
    
    if len(gt_visible) == 0:
        return {
            'f1_max': 0.0,
            'iou_max': 0.0,
            'precision_max': 0.0,
            'recall_max': 0.0,
            'Pd': radar_config.probability_of_detection,
            'Pfa': radar_config.probability_of_false_alarm
        }
    
    # Simulate perfect sensor with realistic limitations
    Pd = radar_config.probability_of_detection  # e.g., 0.9
    Pfa = radar_config.probability_of_false_alarm  # e.g., 0.05
    
    # Create "perfect" detection map (what sensor would return)
    perfect_detection = np.zeros_like(gt_visible)
    
    # True occupied pixels: detect with probability Pd
    occupied_mask = gt_visible > threshold
    perfect_detection[occupied_mask] = Pd
    
    # True empty pixels: false alarm with probability Pfa
    empty_mask = ~occupied_mask
    perfect_detection[empty_mask] = Pfa
    
    # Add spatial blurring (beam width effect - 3° = multiple pixels)
    # This is a simplification of the thick arc effect
    
    # Compute metrics on this "perfect + realistic" map
    pred_bin = perfect_detection > threshold
    gt_bin = gt_visible > threshold
    
    TP = (pred_bin & gt_bin).sum()
    FP = (pred_bin & (~gt_bin)).sum()
    FN = ((~pred_bin) & gt_bin).sum()
    
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # IoU
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    iou = intersection / (union + 1e-7)
    
    return {
        'f1_max': float(f1),
        'iou_max': float(iou),
        'precision_max': float(precision),
        'recall_max': float(recall),
        'Pd': Pd,
        'Pfa': Pfa
    }


def compute_visible_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    gt_visibility_mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics ONLY on physically visible pixels WITHIN FOV
    
    THE "GLASS CEILING" TEST:
    Radar operates in a limited FOV and cannot see through walls/cars.
    This function evaluates only on pixels that are BOTH:
    1. Physically reachable (not occluded by obstacles)
    2. Within radar's FOV (e.g., 120° forward arc)
    
    This defines the "Theoretical Maximum" - the best possible performance
    given the physical constraints of the radar system.
    
    Example: If glass ceiling = 30% of map, and we get:
             - Global F1 = 0.4 (penalized for 70% outside glass ceiling)
             - F1_visible = 0.85 (on the 30% we CAN scan)
             Then we're achieving 85% of theoretical maximum!
    
    Args:
        prediction: (H, W) predicted occupancy probability [0, 1]
        ground_truth: (H, W) true occupancy [0, 1]
        gt_visibility_mask: (H, W) boolean mask (visible AND in FOV)
        threshold: Threshold for binarization (default 0.5)
    
    Returns:
        dict with visible-only metrics and visibility statistics
    """
    # Extract only visible pixels
    visible_pred = prediction[gt_visibility_mask]
    visible_gt = ground_truth[gt_visibility_mask]
    
    if len(visible_gt) == 0:
        return {
            'f1_visible': 0.0,
            'precision_visible': 0.0,
            'recall_visible': 0.0,
            'error_visible': 0.0,
            'iou_visible': 0.0,
            'target_only_error_visible': 0.0,
            'visibility_ratio': 0.0,  # What % of map is visible?
            'occupied_visible_ratio': 0.0  # What % of targets are visible?
        }
    
    # Binary segmentation
    pred_bin = (visible_pred > threshold).astype(float)
    gt_bin = (visible_gt > threshold).astype(float)
    
    # Compute metrics
    TP = (pred_bin * gt_bin).sum()
    FP = (pred_bin * (1 - gt_bin)).sum()
    FN = ((1 - pred_bin) * gt_bin).sum()
    
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # MAE on visible pixels
    error_visible = np.abs(visible_pred - visible_gt).mean()
    
    # IoU on visible pixels
    intersection = (pred_bin * gt_bin).sum()
    union = ((pred_bin + gt_bin) > 0).sum()
    iou = intersection / (union + 1e-7)
    
    # Target-only error on visible targets
    visible_targets = (visible_gt > threshold)
    if visible_targets.sum() > 0:
        target_error_visible = np.abs(visible_pred[visible_targets] - visible_gt[visible_targets]).mean()
    else:
        target_error_visible = 0.0
    
    # Visibility statistics
    total_pixels = ground_truth.size
    visible_pixels = gt_visibility_mask.sum()
    visibility_ratio = visible_pixels / total_pixels
    
    total_occupied = (ground_truth > threshold).sum()
    if total_occupied > 0:
        occupied_visible = (ground_truth[gt_visibility_mask] > threshold).sum()
        occupied_visible_ratio = occupied_visible / total_occupied
    else:
        occupied_visible_ratio = 0.0
    
    return {
        'f1_visible': float(f1),
        'precision_visible': float(precision),
        'recall_visible': float(recall),
        'error_visible': float(error_visible),
        'iou_visible': float(iou),
        'target_only_error_visible': float(target_error_visible),
        'visibility_ratio': float(visibility_ratio),  # e.g., 0.6 = 60% of map visible
        'occupied_visible_ratio': float(occupied_visible_ratio)  # e.g., 0.7 = 70% of targets visible
    }
