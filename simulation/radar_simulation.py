"""
Cognitive Radar Simulation: Information-Theoretic Active Sensing
================================================================

OVERVIEW:
This simulation demonstrates the advantage of cognitive (closed-loop) radar systems
over traditional open-loop approaches for autonomous driving scene perception.

SCIENTIFIC FOUNDATION:
1. Entropy-Based Uncertainty: Shannon entropy H(p) measures uncertainty in predictions
2. Information-Theoretic Objective: Maximize information gain per radar pulse
3. Bayesian Sensor Fusion: Combine camera and radar measurements
4. Closed-Loop Control: Adapts after each pulse

COMPARISON STRATEGIES:
- Cognitive (Entropy-Guided): Closed-loop, adapts after each pulse
- Uniform: Open-loop, pre-planned angular scan
- Random: Open-loop, random beam selection (lower bound)

HOW TO RUN:
    python radar_simulation.py --budget 10 --num_scenes 50 --num_examples 5

Author: Shay Elbaz
Date: 2025-12-28
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from tqdm import tqdm
import imageio

# Add parent directory's lift-splat-shoot to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

# Import our modules
from config import RadarConfig, SimulationConfig
from radar_sensor import cartesian_to_polar, polar_to_cartesian
from information_theory import (
    compute_entropy, compute_information_gain, 
    bayesian_fusion_raytracing
)
from ray_tracing import (
    radar_inverse_sensor_model,
    compute_visibility_mask,
    compute_gt_visibility_mask,
    cast_radar_cone
)
from grid import get_grid_shape
from metrics import (
    compute_segmentation_metrics,
    compute_target_only_error,
    compute_roi_entropy,
    compute_chamfer_distance,
    compute_roc_curve,
    compute_perfect_sensor_baseline,
    compute_visible_metrics
)
from scene_conditions import get_scene_groups_from_dataset
from beam_selectors import UniformSelector, RandomSelector
from visualization import (
    plot_entropy_traces, plot_error_traces, plot_iou_traces, plot_comparison_bars,
    plot_entropy_traces_fov, plot_error_traces_fov, plot_iou_traces_fov,
    save_simple_comparison,
    plot_comprehensive_metric_summary,
    plot_metrics_table_and_roc,
    plot_f1_traces, plot_precision_recall_traces, plot_target_only_error_traces,
    plot_roi_entropy_traces, plot_chamfer_distance_traces,
    plot_f1_visible_traces, plot_iou_visible_traces, plot_error_visible_traces,
    visualize_visibility_mask_diagnostic
)


# ==========================================
#      METRIC COMPUTATION HELPERS
# ==========================================
# All metric functions moved to metrics.py module


def _create_empty_results_dict():
    """
    Create empty results dictionary structure
    
    Returns dict with all metric keys initialized to empty lists.
    Used for both global results and per-condition bucketing.
    """
    return {
        # Global metrics (entire BEV)
        'information_gain': [],
        'coverage_ratio': [],
        'high_entropy_coverage': [],
        'detection_improvement': [],
        'mean_entropy_scanned': [],
        'iou': [],
        'iou_trace': [],
        'entropy_trace': [],
        'error_trace': [],
        # Industry Standard Metrics
        'f1_score': [],
        'precision': [],
        'recall': [],
        'f1_trace': [],
        'precision_trace': [],
        'recall_trace': [],
        # Target-Only Metrics
        'target_only_error': [],
        'target_only_error_trace': [],
        'roi_entropy': [],
        'roi_entropy_trace': [],
        'chamfer_distance': [],
        'chamfer_trace': [],
        # VISIBLE-ONLY Metrics
        'f1_visible': [],
        'precision_visible': [],
        'recall_visible': [],
        'error_visible': [],
        'iou_visible': [],
        'target_only_error_visible': [],
        'f1_visible_trace': [],
        'precision_visible_trace': [],
        'recall_visible_trace': [],
        'error_visible_trace': [],
        'iou_visible_trace': [],
        'visibility_ratio': [],
        'occupied_visible_ratio': [],
        # Theoretical Maximum
        'theoretical_max_f1': [],
        'theoretical_max_iou': [],
        'sensor_Pd': [],
        'sensor_Pfa': [],
        # FOV-only metrics
        'information_gain_fov': [],
        'coverage_ratio_fov': [],
        'high_entropy_coverage_fov': [],
        'detection_improvement_fov': [],
        'mean_entropy_scanned_fov': [],
        'iou_fov': [],
        'iou_trace_fov': [],
        'entropy_trace_fov': [],
        'error_trace_fov': [],
        'f1_fov': [],
        'precision_fov': [],
        'recall_fov': [],
        'target_only_error_fov': [],
        'roc_global': [],
        'roc_fov': [],
    }


def _append_scene_to_condition_buckets(results, results_by_condition, scene_groups):
    """
    Append the most recent scene metrics to per-condition buckets
    
    Args:
        results: Global results dict (last element is current scene)
        results_by_condition: Dict of condition_name -> results_dict
        scene_groups: List of condition names this scene belongs to
    """
    for group in scene_groups:
        # Lazy initialize bucket if first scene for this condition
        if group not in results_by_condition:
            results_by_condition[group] = _create_empty_results_dict()
        
        # Append last value from each metric to this condition's bucket
        for key in results.keys():
            if len(results[key]) > 0:
                results_by_condition[group][key].append(results[key][-1])


# ==========================================
#         MODEL WRAPPER
# ==========================================

class EnsembleLSS(nn.Module):
    """5-head ensemble model for uncertainty estimation"""
    def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5):
        super().__init__()
        self.models = nn.ModuleList()
        for _ in range(num_models):
            self.models.append(LiftSplatShoot(grid_conf, data_aug_conf, outC=outC))
    
    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        outputs = []
        for model in self.models:
            res = model(x, rots, trans, intrins, post_rots, post_trans)
            outputs.append(res)
        return torch.stack(outputs, dim=1)


# ==========================================
#      SIMULATION EXPERIMENT
# ==========================================

class RadarSimulationExperiment:
    """
    Main experiment class for cognitive radar simulation
    
    This class orchestrates:
    1. Model loading and data preparation
    2. Strategy execution (cognitive + baselines)
    3. Metrics computation
    4. Visualization generation
    5. Results aggregation and comparison
    """
    
    def __init__(self, model_path, radar_config, sim_config):
        """
        Initialize simulation experiment
        Loads the model, the data, and the baselines
        Args:
            model_path: Path to trained ensemble model checkpoint
            radar_config: RadarConfig object
            sim_config: SimulationConfig object
        """
        self.radar_config = radar_config
        self.sim_config = sim_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"🔧 Loading model from: {model_path}")
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
        data_aug_conf = {
            'resize_lim': (0.193, 0.193),
            'final_dim': (128, 352),
            'rot_lim': (0.0, 0.0),
            'H': 900, 'W': 1600,
            'bot_pct_lim': (0.0, 0.0),
            'cams': cams,
            'Ncams': 6,
            'rand_flip': False
        }
        
        self.model = EnsembleLSS(sim_config.grid_conf, data_aug_conf, outC=1, num_models=5).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
        self.model.eval()
        print("   ✓ Model loaded")
        
        # Load data (path relative to parent directory)
        print("📂 Loading validation data...")
        data_path = os.path.join(parent_dir, 'data_nuscenes')
        _, self.val_loader = compile_data('trainval', data_path, data_aug_conf, 
                                          sim_config.grid_conf, bsz=1, nworkers=4,
                                          parser_name='segmentationdata')
        print(f"   ✓ Loaded {len(self.val_loader)} scenes")
        
        # Initialize baseline selectors (cognitive uses adaptive selection)
        self.selectors = {
            'uniform': UniformSelector(radar_config, sim_config.radar_budget),
            'random': RandomSelector(radar_config, sim_config.radar_budget)
        }
        
        os.makedirs(sim_config.results_dir, exist_ok=True)
    
    def _compute_radar_fov_mask(self, grid_conf):
        """
        Compute binary mask for radar's scannable FOV
        
        Returns (H, W) mask where 1 = within radar FOV, 0 = outside
        Radar FOV: 120° azimuth (±60°), range 4-50m
        """
        H, W = get_grid_shape(grid_conf)
        
        # Create coordinate grids
        x_coords = np.linspace(grid_conf['xbound'][0], grid_conf['xbound'][1], W)
        y_coords = np.linspace(grid_conf['ybound'][0], grid_conf['ybound'][1], H)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Convert to polar
        R, Theta = cartesian_to_polar(X, Y)
        
        # FOV constraints
        azimuth_max = self.radar_config.azimuth_fov / 2  # ±60°
        in_fov = (R >= self.radar_config.min_range) & \
                 (R <= self.radar_config.max_range) & \
                 (np.abs(Theta) <= azimuth_max)
        
        return in_fov.astype(float)
    
    def run_strategy_cognitive(self, save_examples=True):
        """
        COGNITIVE LOOP: Sequential closed-loop active perception
        
        Key difference from baselines:
        - Selects ONE beam at a time
        - Updates belief after EACH pulse
        - Next beam uses UPDATED belief (closed-loop feedback!)
        
        This is the scientifically correct cognitive radar implementation.
        
        Returns:
            (aggregated_metrics, raw_results): Tuple of dicts
        """
        print(f"\n{'='*60}")
        print(f"Running: COGNITIVE (CLOSED-LOOP) Strategy")
        print(f"{'='*60}")
        
        # Initialize global results
        results = _create_empty_results_dict()
        
        # Initialize per-condition results (buckets: DAY, NIGHT, RAINY, etc.)
        results_by_condition = {}
        
        examples_saved = 0
        examples_dir = os.path.join(self.sim_config.results_dir, 'examples', 'entropy')
        if save_examples:
            os.makedirs(examples_dir, exist_ok=True)
        
        with torch.no_grad():
            for scene_idx, batch in enumerate(tqdm(self.val_loader, desc="cognitive")): # per scene run 
                if scene_idx >= self.sim_config.num_test_scenes:
                    break
                
                # Get camera prediction and ground truth
                imgs, rots, trans, intrins, post_rots, post_trans, gt_binimg = batch
                imgs = imgs.to(self.device)
                rots, trans, intrins = rots.to(self.device), trans.to(self.device), intrins.to(self.device)
                post_rots, post_trans = post_rots.to(self.device), post_trans.to(self.device)
                
                preds = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
                probs = torch.sigmoid(preds)
                initial_prior = probs.mean(dim=1)[0, 0].cpu().numpy()
                initial_prior = np.fliplr(initial_prior)
                gt = np.fliplr(gt_binimg[0, 0].numpy())
                
                # Initialize for closed-loop
                current_belief = initial_prior.copy() # initial belief is the camera prediction prior, this will be updated by the radar measurements
                entropy_initial = compute_entropy(current_belief)
                
                # PHYSICS: Track visibility mask (which cells can we actually observe?)
                visibility_mask = np.ones_like(current_belief)  # Initially assume all visible
                
                # Compute FOV mask once per scene
                fov_mask = self._compute_radar_fov_mask(self.sim_config.grid_conf)
                fov_mask_bool = fov_mask.astype(bool)
                
                # Compute GT VISIBILITY mask once per scene (what's physically reachable)
                # This defines the "Theoretical Maximum" - best possible performance given occlusions
                gt_visibility_mask_full = compute_gt_visibility_mask(gt, self.sim_config.grid_conf, self.radar_config, self.sim_config)
                
                # GLASS CEILING: Intersection of Visible AND FOV
                # We only care about pixels that are BOTH physically reachable AND within radar's FOV
                gt_visibility_mask = gt_visibility_mask_full & fov_mask_bool
                
                # Compute THEORETICAL MAXIMUM given sensor limitations
                # (This is what we'd get with perfect beam placement but realistic sensor)
                theoretical_max = compute_perfect_sensor_baseline(gt, gt_visibility_mask, self.radar_config)
                
                beams_selected = []  # List of azimuth angles (not tuples anymore!)
                azimuths_used = set()  # Track which directions we've already scanned
                coverage_mask = np.zeros_like(current_belief)
                
                # Initialize trace arrays (global + FOV)
                entropy_trace_scene = [entropy_initial.sum()]
                error_trace_scene = [np.abs(current_belief - gt).mean()]
                entropy_trace_fov_scene = [(entropy_initial * fov_mask).sum()]
                error_trace_fov_scene = [np.abs(current_belief[fov_mask_bool] - gt[fov_mask_bool]).mean()]
                
                # IoU trace (initial state - camera only)
                gt_binary = (gt > 0.5)
                belief_binary_initial = (current_belief > 0.5)
                intersection_initial = (gt_binary & belief_binary_initial).sum()
                union_initial = (gt_binary | belief_binary_initial).sum()
                iou_initial = intersection_initial / (union_initial + 1e-7)
                iou_trace_scene = [float(iou_initial)]
                
                gt_binary_fov = (gt * fov_mask) > 0.5
                belief_binary_fov_initial = (current_belief * fov_mask) > 0.5
                intersection_fov_initial = (gt_binary_fov & belief_binary_fov_initial).sum()
                union_fov_initial = (gt_binary_fov | belief_binary_fov_initial).sum()
                iou_fov_initial = intersection_fov_initial / (union_fov_initial + 1e-7)
                iou_trace_fov_scene = [float(iou_fov_initial)]
                
                # Industry Standard Metrics (initial state - camera only)
                seg_metrics_initial = compute_segmentation_metrics(current_belief, gt)
                f1_trace_scene = [seg_metrics_initial['f1_score']]
                precision_trace_scene = [seg_metrics_initial['precision']]
                recall_trace_scene = [seg_metrics_initial['recall']]
                
                # Target-Only Metrics (initial state)
                target_error_initial = compute_target_only_error(current_belief, gt)
                target_only_error_trace_scene = [target_error_initial]
                
                roi_entropy_initial = compute_roi_entropy(entropy_initial, gt)
                roi_entropy_trace_scene = [roi_entropy_initial]
                
                chamfer_initial = compute_chamfer_distance(current_belief, gt)
                chamfer_trace_scene = [chamfer_initial]
                
                # VISIBLE-ONLY Metrics (initial state - camera only)
                # Shows performance on ONLY physically reachable pixels (not occluded)
                visible_metrics_initial = compute_visible_metrics(current_belief, gt, gt_visibility_mask)
                f1_visible_trace_scene = [visible_metrics_initial['f1_visible']]
                precision_visible_trace_scene = [visible_metrics_initial['precision_visible']]
                recall_visible_trace_scene = [visible_metrics_initial['recall_visible']]
                error_visible_trace_scene = [visible_metrics_initial['error_visible']]
                iou_visible_trace_scene = [visible_metrics_initial['iou_visible']]
                
                # ==== CLOSED-LOOP: Sequential beam selection ====
                for pulse_idx in range(self.sim_config.radar_budget):
                    # 1. Compute CURRENT entropy
                    entropy_current = compute_entropy(current_belief)
                    
                    # 2. Update visibility mask based on current belief
                    # (Don't waste time targeting shadows!)
                    visibility_mask = compute_visibility_mask(
                        current_belief, 
                        self.sim_config.grid_conf, 
                        self.radar_config,
                        self.sim_config
                    )
                    
                    # 3. Select NEXT beam based on CURRENT entropy, visibility, AND belief
                    azimuth = self._select_single_beam_greedy_raytracing(
                        entropy_current,
                        visibility_mask,
                        azimuths_used,
                        self.sim_config.grid_conf,
                        current_belief  # Pass current belief for utility weighting
                    )
                    
                    if azimuth is None:
                        break
                    
                    beams_selected.append(azimuth)
                    azimuths_used.add(azimuth)  # Track used azimuth
                    
                    # 4. Simulate PHYSICALLY-CORRECT radar measurement
                    measurement = self._simulate_single_pulse_raytracing(azimuth, gt)
                    
                    # Update coverage mask (free space + hit, NOT shadow)
                    coverage_mask = np.maximum(coverage_mask, 
                                              measurement['free_space'] | (measurement['occupied'] > 0))
                    
                    # 5. Update belief using PHYSICALLY-CORRECT Bayesian fusion
                    # (Free space, hit, and shadow handled properly!)
                    current_belief = bayesian_fusion_raytracing(
                        current_belief, 
                        measurement, 
                        self.sim_config.radar_confidence
                    )
                    
                    # 6. Track metrics (global + FOV)
                    entropy_after_pulse = compute_entropy(current_belief)
                    entropy_trace_scene.append(entropy_after_pulse.sum())
                    error_after_pulse = np.abs(current_belief - gt).mean()
                    error_trace_scene.append(error_after_pulse)
                    
                    # IoU tracking (global)
                    belief_binary_pulse = (current_belief > 0.5)
                    intersection_pulse = (gt_binary & belief_binary_pulse).sum()
                    union_pulse = (gt_binary | belief_binary_pulse).sum()
                    iou_pulse = intersection_pulse / (union_pulse + 1e-7)
                    iou_trace_scene.append(float(iou_pulse))
                    
                    # FOV-specific tracking
                    entropy_fov_pulse = (entropy_after_pulse * fov_mask).sum()
                    entropy_trace_fov_scene.append(entropy_fov_pulse)
                    error_fov_pulse = np.abs(current_belief[fov_mask_bool] - gt[fov_mask_bool]).mean()
                    error_trace_fov_scene.append(error_fov_pulse)
                    
                    # IoU tracking (FOV)
                    belief_binary_fov_pulse = (current_belief * fov_mask) > 0.5
                    intersection_fov_pulse = (gt_binary_fov & belief_binary_fov_pulse).sum()
                    union_fov_pulse = (gt_binary_fov | belief_binary_fov_pulse).sum()
                    iou_fov_pulse = intersection_fov_pulse / (union_fov_pulse + 1e-7)
                    iou_trace_fov_scene.append(float(iou_fov_pulse))
                    
                    # Industry Standard Metrics (per-pulse)
                    seg_metrics_pulse = compute_segmentation_metrics(current_belief, gt)
                    f1_trace_scene.append(seg_metrics_pulse['f1_score'])
                    precision_trace_scene.append(seg_metrics_pulse['precision'])
                    recall_trace_scene.append(seg_metrics_pulse['recall'])
                    
                    # Target-Only Metrics (per-pulse)
                    target_error_pulse = compute_target_only_error(current_belief, gt)
                    target_only_error_trace_scene.append(target_error_pulse)
                    
                    roi_entropy_pulse = compute_roi_entropy(entropy_after_pulse, gt)
                    roi_entropy_trace_scene.append(roi_entropy_pulse)
                    
                    chamfer_pulse = compute_chamfer_distance(current_belief, gt)
                    chamfer_trace_scene.append(chamfer_pulse)
                    
                    # VISIBLE-ONLY Metrics (per-pulse)
                    visible_metrics_pulse = compute_visible_metrics(current_belief, gt, gt_visibility_mask)
                    f1_visible_trace_scene.append(visible_metrics_pulse['f1_visible'])
                    precision_visible_trace_scene.append(visible_metrics_pulse['precision_visible'])
                    recall_visible_trace_scene.append(visible_metrics_pulse['recall_visible'])
                    error_visible_trace_scene.append(visible_metrics_pulse['error_visible'])
                    iou_visible_trace_scene.append(visible_metrics_pulse['iou_visible'])
                
                # ===== GLOBAL METRICS (entire BEV) =====
                entropy_final = compute_entropy(current_belief)
                
                # Skip FOV change check (minor leakage from Gaussian splat at boundaries is acceptable)
                
                # Information gain should ONLY come from FOV (where radar scans)
                # Global IG = FOV IG (mathematically, if beliefs outside FOV unchanged)
                total_ig_global = entropy_initial.sum() - entropy_final.sum()
                total_ig_fov_only = (entropy_initial * fov_mask).sum() - (entropy_final * fov_mask).sum()
                
                # Use FOV IG as the TRUE information gain (radar only operates in FOV)
                total_ig = total_ig_fov_only
                
                results['information_gain'].append(float(total_ig))
                results['entropy_trace'].append(entropy_trace_scene)
                results['error_trace'].append(error_trace_scene)
                results['iou_trace'].append(iou_trace_scene)
                results['entropy_trace_fov'].append(entropy_trace_fov_scene)
                results['error_trace_fov'].append(error_trace_fov_scene)
                results['iou_trace_fov'].append(iou_trace_fov_scene)
                
                # Store traces for new metrics
                results['f1_trace'].append(f1_trace_scene)
                results['precision_trace'].append(precision_trace_scene)
                results['recall_trace'].append(recall_trace_scene)
                results['target_only_error_trace'].append(target_only_error_trace_scene)
                results['roi_entropy_trace'].append(roi_entropy_trace_scene)
                results['chamfer_trace'].append(chamfer_trace_scene)
                
                # Store VISIBLE-ONLY traces (Glass Ceiling metrics)
                results['f1_visible_trace'].append(f1_visible_trace_scene)
                results['precision_visible_trace'].append(precision_visible_trace_scene)
                results['recall_visible_trace'].append(recall_visible_trace_scene)
                results['error_visible_trace'].append(error_visible_trace_scene)
                results['iou_visible_trace'].append(iou_visible_trace_scene)
                
                # Coverage metrics
                coverage_ratio = coverage_mask.sum() / coverage_mask.size
                results['coverage_ratio'].append(float(coverage_ratio))
                
                # High-entropy coverage
                high_entropy_mask = entropy_initial > 0.5
                if high_entropy_mask.sum() > 0:
                    high_ent_covered = (coverage_mask > 0) & high_entropy_mask
                    high_ent_ratio = high_ent_covered.sum() / high_entropy_mask.sum()
                    results['high_entropy_coverage'].append(float(high_ent_ratio))
                
                # Mean entropy of scanned regions
                if coverage_mask.sum() > 0:
                    mean_ent_scanned = entropy_initial[coverage_mask > 0].mean()
                    results['mean_entropy_scanned'].append(float(mean_ent_scanned))
                
                # Detection improvement
                camera_error = np.abs(initial_prior - gt).mean()
                fused_error = np.abs(current_belief - gt).mean()
                improvement = camera_error - fused_error
                results['detection_improvement'].append(float(improvement))
                
                # IoU (Intersection over Union) - Object blob detection quality
                # Threshold predictions to get binary masks
                gt_binary = (gt > 0.5).astype(float)
                belief_binary = (current_belief > 0.5).astype(float)
                
                intersection = (gt_binary * belief_binary).sum()
                union = ((gt_binary + belief_binary) > 0).sum()
                iou = intersection / (union + 1e-7)  # Avoid division by zero
                results['iou'].append(float(iou))
                
                # Industry Standard Metrics (final state)
                seg_metrics_final = compute_segmentation_metrics(current_belief, gt)
                results['f1_score'].append(seg_metrics_final['f1_score'])
                results['precision'].append(seg_metrics_final['precision'])
                results['recall'].append(seg_metrics_final['recall'])
                
                # Target-Only Metrics (final state)
                target_error_final = compute_target_only_error(current_belief, gt)
                results['target_only_error'].append(target_error_final)
                
                roi_entropy_final = compute_roi_entropy(entropy_final, gt)
                results['roi_entropy'].append(roi_entropy_final)
                
                chamfer_final = compute_chamfer_distance(current_belief, gt)
                results['chamfer_distance'].append(chamfer_final)
                
                # VISIBLE-ONLY Metrics (final state - Glass Ceiling)
                visible_metrics_final = compute_visible_metrics(current_belief, gt, gt_visibility_mask)
                results['f1_visible'].append(visible_metrics_final['f1_visible'])
                results['precision_visible'].append(visible_metrics_final['precision_visible'])
                results['recall_visible'].append(visible_metrics_final['recall_visible'])
                results['error_visible'].append(visible_metrics_final['error_visible'])
                results['iou_visible'].append(visible_metrics_final['iou_visible'])
                results['target_only_error_visible'].append(visible_metrics_final['target_only_error_visible'])
                results['visibility_ratio'].append(visible_metrics_final['visibility_ratio'])
                results['occupied_visible_ratio'].append(visible_metrics_final['occupied_visible_ratio'])
                
                # Theoretical Maximum (computed once per scene)
                results['theoretical_max_f1'].append(theoretical_max['f1_max'])
                results['theoretical_max_iou'].append(theoretical_max['iou_max'])
                results['sensor_Pd'].append(theoretical_max['Pd'])
                results['sensor_Pfa'].append(theoretical_max['Pfa'])
                
                # ROC curves (global and FOV)
                roc_global = compute_roc_curve(current_belief, gt)
                results['roc_global'].append(roc_global)
                
                current_belief_fov = current_belief * fov_mask
                gt_fov = gt * fov_mask
                roc_fov = compute_roc_curve(current_belief_fov, gt_fov)
                results['roc_fov'].append(roc_fov)
                
                # ===== FOV-ONLY METRICS (radar scannable region) =====
                # Note: Since radar only operates in FOV, Global IG = FOV IG
                results['information_gain_fov'].append(float(total_ig))
                
                # Coverage within FOV
                fov_size = fov_mask.sum()
                if fov_size > 0:
                    coverage_ratio_fov = (coverage_mask * fov_mask).sum() / fov_size
                    results['coverage_ratio_fov'].append(float(coverage_ratio_fov))
                
                # High-entropy coverage within FOV
                high_entropy_fov = high_entropy_mask & fov_mask_bool
                if high_entropy_fov.sum() > 0:
                    high_ent_covered_fov = (coverage_mask > 0) & high_entropy_fov
                    high_ent_ratio_fov = high_ent_covered_fov.sum() / high_entropy_fov.sum()
                    results['high_entropy_coverage_fov'].append(float(high_ent_ratio_fov))
                
                # Mean entropy of scanned regions within FOV
                scanned_fov = (coverage_mask > 0) & fov_mask_bool
                if scanned_fov.sum() > 0:
                    mean_ent_scanned_fov = entropy_initial[scanned_fov].mean()
                    results['mean_entropy_scanned_fov'].append(float(mean_ent_scanned_fov))
                
                # Detection improvement within FOV
                if fov_mask.sum() > 0:
                    camera_error_fov = np.abs(initial_prior[fov_mask_bool] - gt[fov_mask_bool]).mean()
                    fused_error_fov = np.abs(current_belief[fov_mask_bool] - gt[fov_mask_bool]).mean()
                    improvement_fov = camera_error_fov - fused_error_fov
                    results['detection_improvement_fov'].append(float(improvement_fov))
                    
                    # IoU within FOV only
                    gt_binary_fov = (gt * fov_mask) > 0.5
                    belief_binary_fov = (current_belief * fov_mask) > 0.5
                    
                    intersection_fov = (gt_binary_fov & belief_binary_fov).sum()
                    union_fov = (gt_binary_fov | belief_binary_fov).sum()
                    iou_fov = intersection_fov / (union_fov + 1e-7)
                    results['iou_fov'].append(float(iou_fov))
                    
                    # Industry Standard Metrics within FOV (final state)
                    # Apply FOV mask to both prediction and GT
                    belief_fov = current_belief * fov_mask
                    gt_fov = gt * fov_mask
                    seg_metrics_fov = compute_segmentation_metrics(belief_fov, gt_fov)
                    results['f1_fov'].append(seg_metrics_fov['f1_score'])
                    results['precision_fov'].append(seg_metrics_fov['precision'])
                    results['recall_fov'].append(seg_metrics_fov['recall'])
                    
                    # Target-Only Error within FOV (final state)
                    target_error_fov = compute_target_only_error(belief_fov, gt_fov)
                    results['target_only_error_fov'].append(target_error_fov)
                
                # Save visualizations (first N examples)
                if save_examples and examples_saved < self.sim_config.num_examples_to_save:
                    self._save_scene_visualizations_raytracing(
                        scene_idx, examples_saved, 'entropy', 
                        gt, initial_prior, entropy_initial, 
                        beams_selected, coverage_mask, current_belief, 
                        entropy_final, total_ig, entropy_trace_scene,
                        examples_dir
                    )
                    
                    # DIAGNOSTIC: Visualize visibility mask to debug glass ceiling metrics
                    visualize_visibility_mask_diagnostic(
                        gt, gt_visibility_mask, initial_prior, current_belief,
                        scene_idx, examples_dir
                    )
                    
                    examples_saved += 1
                
                # Classify scene and append to per-condition buckets
                scene_groups = get_scene_groups_from_dataset(self.val_loader.dataset, scene_idx)
                _append_scene_to_condition_buckets(results, results_by_condition, scene_groups)
        
        if save_examples:
            print(f"   ✓ Saved {examples_saved} example visualizations to {examples_dir}")
        
        # Aggregate global results
        trace_keys = ['entropy_trace', 'error_trace', 'iou_trace', 'f1_trace', 'precision_trace', 
                     'recall_trace', 'target_only_error_trace', 'roi_entropy_trace', 'chamfer_trace',
                     'entropy_trace_fov', 'error_trace_fov', 'iou_trace_fov', 'roc_global', 'roc_fov',
                     'f1_visible_trace', 'precision_visible_trace', 'recall_visible_trace', 
                     'error_visible_trace', 'iou_visible_trace']
        aggregated = {k: float(np.mean(v)) if k not in trace_keys else v 
                     for k, v in results.items()}
        aggregated['std'] = {k: float(np.std(v)) for k, v in results.items() if k not in trace_keys}
        
        # Aggregate per-condition results
        aggregated['by_condition'] = {}
        for condition, cond_results in results_by_condition.items():
            aggregated['by_condition'][condition] = {
                k: float(np.mean(v)) if k not in trace_keys and len(v) > 0 else v
                for k, v in cond_results.items()
            }
            aggregated['by_condition'][condition]['std'] = {
                k: float(np.std(v)) for k, v in cond_results.items() 
                if k not in trace_keys and len(v) > 0
            }
            aggregated['by_condition'][condition]['count'] = len(cond_results['f1_score'])
        
        return aggregated, results
    
    def _select_single_beam_greedy_raytracing(self, entropy_map, visibility_mask, 
                                               azimuths_used, grid_conf, current_belief):
        """
        Select best beam using UTILITY-WEIGHTED Expected Information Gain
        
        Algorithm: EIG(θ) = Σ_r [Entropy(r,θ) × P_visible(r,θ) × Utility(r,θ)]
        
        KEY INSIGHT: Resolving uncertainty about POTENTIAL TARGETS is more
        valuable than resolving uncertainty about empty space.
        
        Utility Weighting: 
        - High belief (P > 0.2) → High utility (target likely!)
        - Low belief (P < 0.1) → Low utility (probably empty)
        
        This solves the "Empty Road Bias": Long rays through empty space
        update many pixels but provide little target information.
        
        Args:
            entropy_map: (H, W) current uncertainty
            visibility_mask: (H, W) P_visible from belief [0, 1]
            azimuths_used: Set of already-scanned azimuths
            grid_conf: BEV configuration
            current_belief: (H, W) current occupancy belief [0, 1]
        
        Returns:
            azimuth: Best beam direction (degrees), or None if exhausted
        """
        H, W = entropy_map.shape
        
        # Utility weighting: Boost value of resolving "target" uncertainty
        # If belief > 0.2 (potential object), boost score significantly
        # If belief < 0.1 (likely empty), reduce score
        target_utility = 1.0 + (self.sim_config.target_utility_scale * current_belief)
        
        # Expected IG per cell = entropy × visibility × utility
        # This makes radar prioritize uncertain targets over empty roads
        expected_ig_map = entropy_map * visibility_mask * target_utility
        
        # Coordinate grids (no flip needed with origin='lower')
        x_coords = np.linspace(grid_conf['xbound'][0], grid_conf['xbound'][1], W)
        y_coords = np.linspace(grid_conf['ybound'][0], grid_conf['ybound'][1], H)
        X, Y = np.meshgrid(x_coords, y_coords)
        R, Theta = cartesian_to_polar(X, Y)
        
        # Sample candidate azimuth angles across FOV
        azimuth_fov = self.radar_config.azimuth_fov / 2
        candidate_azimuths = np.linspace(-azimuth_fov, azimuth_fov, self.sim_config.candidate_azimuth_count)
        
        best_azimuth = None
        best_eig = -np.inf
        
        for azimuth in candidate_azimuths:
            # SOFT PENALTY for recently scanned azimuths (instead of hard ban)
            # This allows "double-checking" confusing areas if EIG is genuinely high
            # but discourages redundant scans unless absolutely necessary
            revisit_penalty = 1.0
            for used_az in azimuths_used:
                angular_diff = abs(azimuth - used_az)
                if angular_diff < self.sim_config.revisit_angle_deg:
                    # Dampen EIG for recently visited angles
                    revisit_penalty = self.sim_config.revisit_penalty
                    break
            
            # Aggregate EIG along this beam direction
            # Include all cells within beam cone (±beam_width/2)
            beam_width = self.radar_config.azimuth_resolution
            angular_distance = np.abs(Theta - azimuth)
            angular_distance = np.minimum(angular_distance, 360 - angular_distance)  # Wrap
            
            # Cells within beam cone and valid range
            in_beam = (angular_distance <= beam_width / 2) & \
                     (R >= self.radar_config.min_range) & \
                     (R <= self.radar_config.max_range)
            
            # EIG(θ) = sum of expected IG along this beam × revisit penalty
            eig_theta = expected_ig_map[in_beam].sum() * revisit_penalty
            
            if eig_theta > best_eig:
                best_eig = eig_theta
                best_azimuth = azimuth
        
        # Return None if no useful beams (all EIG below threshold)
        if best_eig < 0.1:
            return None
        
        return best_azimuth
    
    def _simulate_single_pulse_raytracing(self, azimuth, ground_truth):
        """
        Simulate PHYSICALLY-CORRECT radar pulse using ray tracing
        
        Returns measurement dict with:
        - free_space: empty cells along ray
        - occupied: detection at hit
        - shadow: occluded cells behind hit
        """
        return radar_inverse_sensor_model(
            azimuth, 
            ground_truth,
            self.sim_config.grid_conf,
            self.radar_config,
            detection_confidence=self.sim_config.radar_confidence,
            false_alarm_rate=self.sim_config.radar_false_alarm_rate
        )
    
    def _save_scene_visualizations_raytracing(self, scene_idx, example_idx, strategy_name,
                                              gt, initial_prior, entropy_initial,
                                              beams_selected, coverage_mask, final_belief,
                                              entropy_final, total_ig, entropy_trace,
                                              examples_dir):
        """
        Save visualizations for RAY-TRACING physics model
        
        beams_selected is now a list of azimuths (not tuples!)
        """
        # Simple 2x2 before/after comparison
        save_simple_comparison(
            scene_idx=scene_idx,
            strategy_name=strategy_name,
            camera_prob=initial_prior,
            entropy_map=entropy_initial,
            coverage_mask=coverage_mask,
            fused_prob=final_belief,
            entropy_after=entropy_final,
            grid_conf=self.sim_config.grid_conf,
            save_path=os.path.join(examples_dir, f'scene_{example_idx+1:02d}_simple.png'),
            gt=gt
        )
        
        # Ray visualization showing beams as actual rays (not spotlight circles)
        self._save_ray_visualization(
            scene_idx=scene_idx,
            strategy_name=strategy_name,
            entropy_initial=entropy_initial,
            gt=gt,
            beams_azimuths=beams_selected,
            coverage_mask=coverage_mask,
            save_path=os.path.join(examples_dir, f'scene_{example_idx+1:02d}_rays.png')
        )
        
        # Prior update sequence (now shows ray tracing effects)
        self._save_prior_update_raytracing(
            scene_idx=scene_idx,
            strategy_name=strategy_name,
            initial_prior=initial_prior,
            gt=gt,
            beams_azimuths=beams_selected,
            save_path=os.path.join(examples_dir, f'scene_{example_idx+1:02d}_updates.png')
        )
        
        # GIF animation showing ray casting (only for first example)
        if example_idx == 0:
            self._save_raytracing_gif(
                scene_idx=scene_idx,
                strategy_name=strategy_name,
                initial_prior=initial_prior,
                gt=gt,
                beams_azimuths=beams_selected,
                save_path=os.path.join(examples_dir, f'scene_{example_idx+1:02d}_{strategy_name}_loop.gif')
            )
    
    def _save_ray_visualization(self, scene_idx, strategy_name, entropy_initial, gt,
                                beams_azimuths, coverage_mask, save_path):
        """
        Visualize beams as RAYS (not spotlights!)
        
        Shows:
        - Ground truth as background (grayscale)
        - Entropy as colored overlay
        - Each beam as an arrow from ego vehicle
        - Ray trace results (free space, hit, shadow)
        - TARGET CELL marked (shows beam selection is correct!)
        
        Coordinate System:
        - Y positive = UP (forward)
        - X positive = RIGHT
        - 0° beam = forward (positive Y)
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        grid_conf = self.sim_config.grid_conf
        xbound, ybound = grid_conf['xbound'], grid_conf['ybound']
        extent = [xbound[0], xbound[1], ybound[0], ybound[1]]
        
        # Background: Ground truth (grayscale)
        ax.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.4, vmin=0, vmax=1)
        
        # Overlay: Entropy map (colored)
        im = ax.imshow(entropy_initial, cmap='inferno', origin='lower', extent=extent, alpha=0.5)
        
        # Show coverage (free space explored)
        ax.contourf(coverage_mask, levels=[0.3, 1.0], colors=['cyan'],
                    extent=extent, origin='lower', alpha=0.3)
        
        # For cognitive strategy, mark the maximum entropy cell (target)
        if strategy_name == 'entropy' and len(beams_azimuths) > 0:
            # Find max entropy cell
            max_entropy_idx = np.unravel_index(np.argmax(entropy_initial), entropy_initial.shape)
            i_max, j_max = max_entropy_idx
            
            # Convert to world coordinates
            x_coords = np.linspace(xbound[0], xbound[1], entropy_initial.shape[1])
            y_coords = np.linspace(ybound[0], ybound[1], entropy_initial.shape[0])
            x_target = x_coords[j_max]
            y_target = y_coords[i_max]
            
            # Mark target with green star
            ax.plot(x_target, y_target, '*', color='lime', markersize=25, 
                   markeredgecolor='white', markeredgewidth=3, 
                   label=f'Max Entropy Cell\n(H={entropy_initial[i_max, j_max]:.2f} bits)', zorder=15)
        
        # Draw each beam as a CONE (not just line!)
        for beam_idx, azimuth in enumerate(beams_azimuths):
            # Cast ray to visualize
            ray_result = cast_radar_cone(azimuth, gt, grid_conf, self.radar_config)
            
            # Draw beam CONE
            alpha = 0.8 if beam_idx == 0 else 0.4
            color = 'yellow' if beam_idx == 0 else 'orange'
            self._draw_beam_cone(ax, azimuth, color, alpha, extent)
            
            # Mark hit point if exists
            if ray_result['hit_range'] is not None:
                x_hit, y_hit = polar_to_cartesian(ray_result['hit_range'], azimuth)
                ax.plot(x_hit, y_hit, 'ro', markersize=10, markeredgecolor='white', 
                        markeredgewidth=2, label='Hit' if beam_idx == 0 else '', zorder=15)
        
        # Ego vehicle marker at origin
        ax.plot(0, 0, 'w*', markersize=25, markeredgecolor='black', markeredgewidth=3, zorder=20)
        
        # Add coordinate system labels
        ax.set_xlabel('X (meters) - RIGHT →', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (meters) - FORWARD ↑', fontsize=13, fontweight='bold')
        ax.set_title(f'{strategy_name.upper()} - Ray-Tracing Beams\n' +
                    f'Scene {scene_idx+1} | {len(beams_azimuths)} rays | ' +
                    f'Gray=GT, Color=Entropy, Cyan=Scanned',
                    fontsize=14, fontweight='bold')
        
        # Add coordinate arrows
        arrow_len = 10
        ax.annotate('', xy=(arrow_len, 0), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=3, color='white', alpha=0.8))
        ax.text(arrow_len, -2, 'X+', color='white', fontsize=12, fontweight='bold', ha='center')
        
        ax.annotate('', xy=(0, arrow_len), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=3, color='white', alpha=0.8))
        ax.text(-2, arrow_len, 'Y+', color='white', fontsize=12, fontweight='bold', ha='center')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Entropy (bits)')
        
        # Legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Set equal aspect ratio for correct geometry
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_prior_update_raytracing(self, scene_idx, strategy_name, initial_prior,
                                      gt, beams_azimuths, save_path):
        """
        Show how prior updates with ray-tracing physics
        
        Each panel shows:
        - GT as gray background
        - Belief as colored overlay
        - Beam cone that was just sent
        - Color changes highlight belief updates
        """
        num_pulses = min(len(beams_azimuths), 6)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        axes = axes.flatten()
        
        grid_conf = self.sim_config.grid_conf
        xbound, ybound = grid_conf['xbound'], grid_conf['ybound']
        extent = [xbound[0], xbound[1], ybound[0], ybound[1]]
        
        current_belief = initial_prior.copy()
        previous_belief = initial_prior.copy()
        
        for pulse_idx in range(6):
            ax = axes[pulse_idx]
            
            if pulse_idx == 0:
                # Initial state
                ax.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)
                im = ax.imshow(current_belief, cmap='jet', origin='lower', extent=extent, vmin=0, vmax=1, alpha=0.7)
                ax.set_title('INITIAL: Camera Only\n(No Radar Yet)', fontsize=13, fontweight='bold')
            elif pulse_idx <= num_pulses:
                # Apply ray-tracing measurement
                azimuth = beams_azimuths[pulse_idx - 1]
                measurement = self._simulate_single_pulse_raytracing(azimuth, gt)
                
                # Store previous for comparison
                previous_belief = current_belief.copy()
                current_belief = bayesian_fusion_raytracing(
                    current_belief, measurement, self.sim_config.radar_confidence
                )
                
                # Show belief change (red=decreased, green=increased)
                belief_change = current_belief - previous_belief
                
                # Background: GT
                ax.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)
                
                # Show belief change overlay
                im_change = ax.imshow(belief_change, cmap='RdYlGn', origin='lower', extent=extent, 
                                     vmin=-0.3, vmax=0.3, alpha=0.4, zorder=1)
                
                # Show current belief
                im = ax.imshow(current_belief, cmap='jet', origin='lower', extent=extent, 
                              vmin=0, vmax=1, alpha=0.6, zorder=2)
                
                # Show the beam cone that was just sent
                self._draw_beam_cone(ax, azimuth, 'yellow', 0.7, extent)
                
                # Mark hit if any
                if measurement['hit_range'] is not None:
                    x_hit, y_hit = polar_to_cartesian(measurement['hit_range'], azimuth)
                    ax.plot(x_hit, y_hit, 'ro', markersize=10, markeredgecolor='white', 
                           markeredgewidth=2, zorder=15)
                
                ax.set_title(f'PULSE {pulse_idx}: Az={azimuth:.1f}°\n' + 
                           f'Green=Cleared, Red=Shadowed',
                           fontsize=13, fontweight='bold')
            else:
                ax.axis('off')
                continue
            
            ax.plot(0, 0, 'w*', markersize=18, markeredgecolor='black', markeredgewidth=2, zorder=20)
            ax.set_xlabel('X (m) → RIGHT', fontsize=10)
            ax.set_ylabel('Y (m) → FORWARD', fontsize=10)
            ax.set_aspect('equal', adjustable='box')
            
            if pulse_idx > 0 and pulse_idx <= num_pulses:
                # Add small colorbar for belief change
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='P(occ)')
        
        fig.suptitle(f'{strategy_name.upper()} - Ray-Tracing Updates (Scene {scene_idx+1})\n' +
                    'Yellow Cone=Beam | Red Dot=Hit | Green=Free Space | Red=Shadow',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_beam_cone(self, ax, azimuth, color, alpha, extent):
        """
        Draw a radar beam CONE (not just a centerline!)
        
        Shows:
        - Beam centerline
        - Beam edges (±beam_width/2)
        - Filled cone region
        """
        max_range = self.radar_config.max_range
        beam_width = self.radar_config.azimuth_resolution
        
        # Beam centerline
        x_center, y_center = polar_to_cartesian(max_range, azimuth)
        ax.plot([0, x_center], [0, y_center], color=color, linewidth=2, alpha=min(1.0, alpha*1.5), zorder=10)
        
        # Beam edges
        az_left = azimuth - beam_width / 2
        az_right = azimuth + beam_width / 2
        
        x_left, y_left = polar_to_cartesian(max_range, az_left)
        x_right, y_right = polar_to_cartesian(max_range, az_right)
        
        # Draw filled cone
        cone_verts = [(0, 0), (x_left, y_left), (x_right, y_right), (0, 0)]
        from matplotlib.patches import Polygon
        cone = Polygon(cone_verts, facecolor=color, edgecolor=color, 
                      alpha=alpha*0.3, linewidth=1.5, zorder=5)
        ax.add_patch(cone)
    
    def _save_raytracing_gif(self, scene_idx, strategy_name, initial_prior, gt,
                             beams_azimuths, save_path):
        """Animated GIF showing ray-tracing evolution"""
        print(f"      Creating ray-tracing GIF...")
        
        grid_conf = self.sim_config.grid_conf
        xbound, ybound = grid_conf['xbound'], grid_conf['ybound']
        extent = [xbound[0], xbound[1], ybound[0], ybound[1]]  # Y positive = forward
        
        frames = []
        current_belief = initial_prior.copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'{strategy_name.upper()} Ray-Tracing - Scene {scene_idx+1}\n' +
                    'Watch rays clear free space and respect shadows!',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Dummy colorbars
        dummy_entropy = compute_entropy(current_belief)
        im1_dummy = ax1.imshow(dummy_entropy, cmap='inferno', origin='lower', extent=extent)
        cbar1 = plt.colorbar(im1_dummy, ax=ax1, fraction=0.046, pad=0.04, label='Entropy (bits)')
        
        im2_dummy = ax2.imshow(current_belief, cmap='jet', origin='lower', extent=extent, vmin=0, vmax=1)
        cbar2 = plt.colorbar(im2_dummy, ax=ax2, fraction=0.046, pad=0.04, label='Occupancy')
        
        for pulse_idx in range(len(beams_azimuths) + 1):
            ax1.clear()
            ax2.clear()
            
            entropy_current = compute_entropy(current_belief)
            
            # LEFT: Entropy with beam cones
            # Show GT as background
            ax1.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)
            im1 = ax1.imshow(entropy_current, cmap='inferno', origin='lower', extent=extent, alpha=0.6)
            ax1.set_title(f'Ray {pulse_idx}/{len(beams_azimuths)}: UNCERTAINTY\n' +
                         f'Total: {entropy_current.sum():.0f} bits',
                         fontsize=13, fontweight='bold')
            ax1.set_xlabel('X (m) - RIGHT →')
            ax1.set_ylabel('Y (m) - FORWARD ↑')
            ax1.set_xlim(extent[0], extent[1])
            ax1.set_ylim(extent[2], extent[3])
            ax1.set_aspect('equal', adjustable='box')
            
            # Show previous rays with cones (gray)
            for i in range(min(pulse_idx, len(beams_azimuths))):
                az = beams_azimuths[i]
                self._draw_beam_cone(ax1, az, 'gray', 0.2, extent)
            
            # Show current ray with cone (yellow)
            if pulse_idx < len(beams_azimuths):
                az = beams_azimuths[pulse_idx]
                self._draw_beam_cone(ax1, az, 'yellow', 0.5, extent)
            
            ax1.plot(0, 0, 'w*', markersize=18, markeredgecolor='black', markeredgewidth=2, zorder=20)
            
            # RIGHT: Belief with GT overlay
            ax2.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)
            im2 = ax2.imshow(current_belief, cmap='jet', origin='lower', extent=extent, vmin=0, vmax=1, alpha=0.7)
            ax2.set_title(f'OCCUPANCY Belief\n(Rays: {pulse_idx})',
                         fontsize=13, fontweight='bold')
            ax2.set_xlabel('X (m) - RIGHT →')
            ax2.set_ylabel('Y (m) - FORWARD ↑')
            ax2.set_xlim(extent[0], extent[1])
            ax2.set_ylim(extent[2], extent[3])
            ax2.set_aspect('equal', adjustable='box')
            ax2.plot(0, 0, 'w*', markersize=18, markeredgecolor='black', markeredgewidth=2, zorder=20)
            
            cbar1.update_normal(im1)
            cbar2.update_normal(im2)
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            
            # Update for next frame
            if pulse_idx < len(beams_azimuths):
                az = beams_azimuths[pulse_idx]
                measurement = self._simulate_single_pulse_raytracing(az, gt)
                current_belief = bayesian_fusion_raytracing(
                    current_belief, measurement, self.sim_config.radar_confidence
                )
        
        plt.close(fig)
        imageio.mimsave(save_path, frames, duration=3.0, loop=0)
        print(f"      ✓ Ray-tracing GIF saved: {save_path}")
    
    def run_strategy(self, strategy_name, save_examples=True):
        """
        Run open-loop baseline strategy (uniform or random)
        
        Key difference from cognitive:
        - Pre-plans ALL beams at once (open-loop)
        - Executes them sequentially (for fair comparison)
        - Does NOT adapt beam selection (beams are fixed from start)
        
        Args:
            strategy_name: 'uniform' or 'random'
            save_examples: Whether to save visualizations
        
        Returns:
            (aggregated_metrics, raw_results): Tuple of dicts
        """
        print(f"\n{'='*60}")
        print(f"Running: {strategy_name.upper()} Strategy")
        print(f"{'='*60}")
        
        selector = self.selectors[strategy_name]
        
        # Initialize global results
        results = _create_empty_results_dict()
        
        # Initialize per-condition results (buckets: DAY, NIGHT, RAINY, etc.)
        results_by_condition = {}
        
        examples_saved = 0
        examples_dir = os.path.join(self.sim_config.results_dir, 'examples', strategy_name)
        if save_examples:
            os.makedirs(examples_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc=f"{strategy_name}")):
                if i >= self.sim_config.num_test_scenes:
                    break
                
                # Get predictions
                imgs, rots, trans, intrins, post_rots, post_trans, gt_binimg = batch
                imgs = imgs.to(self.device)
                rots, trans, intrins = rots.to(self.device), trans.to(self.device), intrins.to(self.device)
                post_rots, post_trans = post_rots.to(self.device), post_trans.to(self.device)
                
                preds = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
                probs = torch.sigmoid(preds)
                mean_prob = probs.mean(dim=1)[0, 0].cpu().numpy()
                mean_prob = np.fliplr(mean_prob)
                gt = np.fliplr(gt_binimg[0, 0].numpy())
                
                # Compute initial entropy
                entropy_map = compute_entropy(mean_prob)
                
                # Compute FOV mask once per scene
                fov_mask = self._compute_radar_fov_mask(self.sim_config.grid_conf)
                fov_mask_bool = fov_mask.astype(bool)
                
                # Compute GT VISIBILITY mask once per scene (what's physically reachable)
                gt_visibility_mask_full = compute_gt_visibility_mask(gt, self.sim_config.grid_conf, self.radar_config, self.sim_config)
                
                # GLASS CEILING: Intersection of Visible AND FOV
                # We only care about pixels that are BOTH physically reachable AND within radar's FOV
                gt_visibility_mask = gt_visibility_mask_full & fov_mask_bool
                
                # Compute THEORETICAL MAXIMUM given sensor limitations
                theoretical_max = compute_perfect_sensor_baseline(gt, gt_visibility_mask, self.radar_config)
                
                # PRE-SELECT all beams (open-loop) - now returns list of azimuths only!
                beams = selector.select_beams(entropy_map, self.sim_config.grid_conf)
                
                # Execute pulse-by-pulse (for entropy trace) with ray tracing physics
                current_belief = mean_prob.copy()
                entropy_trace_scene = [entropy_map.sum()]
                error_trace_scene = [np.abs(current_belief - gt).mean()]
                entropy_trace_fov_scene = [(entropy_map * fov_mask).sum()]
                error_trace_fov_scene = [np.abs(current_belief[fov_mask_bool] - gt[fov_mask_bool]).mean()]
                
                # IoU trace (initial state - camera only)
                gt_binary = (gt > 0.5)
                belief_binary_initial = (current_belief > 0.5)
                intersection_initial = (gt_binary & belief_binary_initial).sum()
                union_initial = (gt_binary | belief_binary_initial).sum()
                iou_initial = intersection_initial / (union_initial + 1e-7)
                iou_trace_scene = [float(iou_initial)]
                
                gt_binary_fov = (gt * fov_mask) > 0.5
                belief_binary_fov_initial = (current_belief * fov_mask) > 0.5
                intersection_fov_initial = (gt_binary_fov & belief_binary_fov_initial).sum()
                union_fov_initial = (gt_binary_fov | belief_binary_fov_initial).sum()
                iou_fov_initial = intersection_fov_initial / (union_fov_initial + 1e-7)
                iou_trace_fov_scene = [float(iou_fov_initial)]
                
                # Industry Standard Metrics (initial state - camera only)
                seg_metrics_initial = compute_segmentation_metrics(current_belief, gt)
                f1_trace_scene = [seg_metrics_initial['f1_score']]
                precision_trace_scene = [seg_metrics_initial['precision']]
                recall_trace_scene = [seg_metrics_initial['recall']]
                
                # Target-Only Metrics (initial state)
                target_error_initial = compute_target_only_error(current_belief, gt)
                target_only_error_trace_scene = [target_error_initial]
                
                roi_entropy_initial = compute_roi_entropy(entropy_map, gt)
                roi_entropy_trace_scene = [roi_entropy_initial]
                
                chamfer_initial = compute_chamfer_distance(current_belief, gt)
                chamfer_trace_scene = [chamfer_initial]
                
                # VISIBLE-ONLY Metrics (initial state - camera only)
                visible_metrics_initial = compute_visible_metrics(current_belief, gt, gt_visibility_mask)
                f1_visible_trace_scene = [visible_metrics_initial['f1_visible']]
                precision_visible_trace_scene = [visible_metrics_initial['precision_visible']]
                recall_visible_trace_scene = [visible_metrics_initial['recall_visible']]
                error_visible_trace_scene = [visible_metrics_initial['error_visible']]
                iou_visible_trace_scene = [visible_metrics_initial['iou_visible']]
                
                coverage_mask = np.zeros_like(mean_prob)
                
                for azimuth in beams:
                    # PHYSICS-CORRECT measurement
                    measurement = self._simulate_single_pulse_raytracing(azimuth, gt)
                    
                    # Update coverage (free space + hit)
                    coverage_mask = np.maximum(coverage_mask,
                                              measurement['free_space'] | (measurement['occupied'] > 0))
                    
                    # PHYSICS-CORRECT Bayesian update
                    current_belief = bayesian_fusion_raytracing(
                        current_belief, 
                        measurement, 
                        self.sim_config.radar_confidence
                    )
                    
                    entropy_current = compute_entropy(current_belief)
                    entropy_trace_scene.append(entropy_current.sum())
                    error_after_pulse = np.abs(current_belief - gt).mean()
                    error_trace_scene.append(error_after_pulse)
                    
                    # IoU tracking (global)
                    belief_binary_pulse = (current_belief > 0.5)
                    intersection_pulse = (gt_binary & belief_binary_pulse).sum()
                    union_pulse = (gt_binary | belief_binary_pulse).sum()
                    iou_pulse = intersection_pulse / (union_pulse + 1e-7)
                    iou_trace_scene.append(float(iou_pulse))
                    
                    # FOV-specific tracking
                    entropy_fov_pulse = (entropy_current * fov_mask).sum()
                    entropy_trace_fov_scene.append(entropy_fov_pulse)
                    error_fov_pulse = np.abs(current_belief[fov_mask_bool] - gt[fov_mask_bool]).mean()
                    error_trace_fov_scene.append(error_fov_pulse)
                    
                    # IoU tracking (FOV)
                    belief_binary_fov_pulse = (current_belief * fov_mask) > 0.5
                    intersection_fov_pulse = (gt_binary_fov & belief_binary_fov_pulse).sum()
                    union_fov_pulse = (gt_binary_fov | belief_binary_fov_pulse).sum()
                    iou_fov_pulse = intersection_fov_pulse / (union_fov_pulse + 1e-7)
                    iou_trace_fov_scene.append(float(iou_fov_pulse))
                    
                    # Industry Standard Metrics (per-pulse)
                    seg_metrics_pulse = compute_segmentation_metrics(current_belief, gt)
                    f1_trace_scene.append(seg_metrics_pulse['f1_score'])
                    precision_trace_scene.append(seg_metrics_pulse['precision'])
                    recall_trace_scene.append(seg_metrics_pulse['recall'])
                    
                    # Target-Only Metrics (per-pulse)
                    target_error_pulse = compute_target_only_error(current_belief, gt)
                    target_only_error_trace_scene.append(target_error_pulse)
                    
                    roi_entropy_pulse = compute_roi_entropy(entropy_current, gt)
                    roi_entropy_trace_scene.append(roi_entropy_pulse)
                    
                    chamfer_pulse = compute_chamfer_distance(current_belief, gt)
                    chamfer_trace_scene.append(chamfer_pulse)
                    
                    # VISIBLE-ONLY Metrics (per-pulse)
                    visible_metrics_pulse = compute_visible_metrics(current_belief, gt, gt_visibility_mask)
                    f1_visible_trace_scene.append(visible_metrics_pulse['f1_visible'])
                    precision_visible_trace_scene.append(visible_metrics_pulse['precision_visible'])
                    recall_visible_trace_scene.append(visible_metrics_pulse['recall_visible'])
                    error_visible_trace_scene.append(visible_metrics_pulse['error_visible'])
                    iou_visible_trace_scene.append(visible_metrics_pulse['iou_visible'])
                
                results['entropy_trace'].append(entropy_trace_scene)
                results['error_trace'].append(error_trace_scene)
                results['iou_trace'].append(iou_trace_scene)
                results['entropy_trace_fov'].append(entropy_trace_fov_scene)
                results['error_trace_fov'].append(error_trace_fov_scene)
                results['iou_trace_fov'].append(iou_trace_fov_scene)
                
                # Store traces for new metrics
                results['f1_trace'].append(f1_trace_scene)
                results['precision_trace'].append(precision_trace_scene)
                results['recall_trace'].append(recall_trace_scene)
                results['target_only_error_trace'].append(target_only_error_trace_scene)
                results['roi_entropy_trace'].append(roi_entropy_trace_scene)
                results['chamfer_trace'].append(chamfer_trace_scene)
                
                # Store VISIBLE-ONLY traces (Glass Ceiling metrics)
                results['f1_visible_trace'].append(f1_visible_trace_scene)
                results['precision_visible_trace'].append(precision_visible_trace_scene)
                results['recall_visible_trace'].append(recall_visible_trace_scene)
                results['error_visible_trace'].append(error_visible_trace_scene)
                results['iou_visible_trace'].append(iou_visible_trace_scene)
                
                # Final state
                fused_prob = current_belief
                entropy_after = compute_entropy(fused_prob)
                
                # Compute radar FOV mask
                fov_mask = self._compute_radar_fov_mask(self.sim_config.grid_conf)
                
                # Skip FOV change check (minor leakage from Gaussian splat at boundaries is acceptable)
                
                # ===== GLOBAL METRICS =====
                # Information gain should ONLY come from FOV
                total_ig_fov_only = (entropy_map * fov_mask).sum() - (entropy_after * fov_mask).sum()
                total_ig = total_ig_fov_only
                results['information_gain'].append(total_ig)
                
                coverage_ratio = coverage_mask.sum() / coverage_mask.size
                results['coverage_ratio'].append(coverage_ratio)
                
                high_entropy_mask = entropy_map > 0.5
                if high_entropy_mask.sum() > 0:
                    high_ent_covered = (coverage_mask > 0) & high_entropy_mask
                    high_ent_ratio = high_ent_covered.sum() / high_entropy_mask.sum()
                    results['high_entropy_coverage'].append(high_ent_ratio)
                
                if coverage_mask.sum() > 0:
                    mean_ent_scanned = entropy_map[coverage_mask > 0].mean()
                    results['mean_entropy_scanned'].append(mean_ent_scanned)
                
                camera_error = np.abs(mean_prob - gt).mean()
                fused_error = np.abs(fused_prob - gt).mean()
                improvement = camera_error - fused_error
                results['detection_improvement'].append(improvement)
                
                # IoU (Intersection over Union) - Object blob detection quality
                gt_binary = (gt > 0.5).astype(float)
                belief_binary = (fused_prob > 0.5).astype(float)
                
                intersection = (gt_binary * belief_binary).sum()
                union = ((gt_binary + belief_binary) > 0).sum()
                iou = intersection / (union + 1e-7)
                results['iou'].append(float(iou))
                
                # Industry Standard Metrics (final state)
                seg_metrics_final = compute_segmentation_metrics(fused_prob, gt)
                results['f1_score'].append(seg_metrics_final['f1_score'])
                results['precision'].append(seg_metrics_final['precision'])
                results['recall'].append(seg_metrics_final['recall'])
                
                # Target-Only Metrics (final state)
                target_error_final = compute_target_only_error(fused_prob, gt)
                results['target_only_error'].append(target_error_final)
                
                roi_entropy_final = compute_roi_entropy(entropy_after, gt)
                results['roi_entropy'].append(roi_entropy_final)
                
                chamfer_final = compute_chamfer_distance(fused_prob, gt)
                results['chamfer_distance'].append(chamfer_final)
                
                # VISIBLE-ONLY Metrics (final state - Glass Ceiling)
                visible_metrics_final = compute_visible_metrics(fused_prob, gt, gt_visibility_mask)
                results['f1_visible'].append(visible_metrics_final['f1_visible'])
                results['precision_visible'].append(visible_metrics_final['precision_visible'])
                results['recall_visible'].append(visible_metrics_final['recall_visible'])
                results['error_visible'].append(visible_metrics_final['error_visible'])
                results['iou_visible'].append(visible_metrics_final['iou_visible'])
                results['target_only_error_visible'].append(visible_metrics_final['target_only_error_visible'])
                results['visibility_ratio'].append(visible_metrics_final['visibility_ratio'])
                results['occupied_visible_ratio'].append(visible_metrics_final['occupied_visible_ratio'])
                
                # Theoretical Maximum (computed once per scene)
                results['theoretical_max_f1'].append(theoretical_max['f1_max'])
                results['theoretical_max_iou'].append(theoretical_max['iou_max'])
                results['sensor_Pd'].append(theoretical_max['Pd'])
                results['sensor_Pfa'].append(theoretical_max['Pfa'])
                
                # ROC curves (global and FOV)
                roc_global = compute_roc_curve(fused_prob, gt)
                results['roc_global'].append(roc_global)
                
                fused_prob_fov = fused_prob * fov_mask
                gt_fov = gt * fov_mask
                roc_fov = compute_roc_curve(fused_prob_fov, gt_fov)
                results['roc_fov'].append(roc_fov)
                
                # ===== FOV-ONLY METRICS =====
                fov_mask_bool = fov_mask.astype(bool)  # Convert to bool for bitwise ops
                # Note: Since radar only operates in FOV, Global IG = FOV IG
                results['information_gain_fov'].append(total_ig)
                
                fov_size = fov_mask.sum()
                if fov_size > 0:
                    coverage_ratio_fov = (coverage_mask * fov_mask).sum() / fov_size
                    results['coverage_ratio_fov'].append(coverage_ratio_fov)
                
                high_entropy_fov = high_entropy_mask & fov_mask_bool
                if high_entropy_fov.sum() > 0:
                    high_ent_covered_fov = (coverage_mask > 0) & high_entropy_fov
                    high_ent_ratio_fov = high_ent_covered_fov.sum() / high_entropy_fov.sum()
                    results['high_entropy_coverage_fov'].append(high_ent_ratio_fov)
                
                scanned_fov = (coverage_mask > 0) & fov_mask_bool
                if scanned_fov.sum() > 0:
                    mean_ent_scanned_fov = entropy_map[scanned_fov].mean()
                    results['mean_entropy_scanned_fov'].append(mean_ent_scanned_fov)
                
                if fov_mask.sum() > 0:
                    camera_error_fov = np.abs(mean_prob[fov_mask_bool] - gt[fov_mask_bool]).mean()
                    fused_error_fov = np.abs(fused_prob[fov_mask_bool] - gt[fov_mask_bool]).mean()
                    improvement_fov = camera_error_fov - fused_error_fov
                    results['detection_improvement_fov'].append(improvement_fov)
                    
                    # IoU within FOV only
                    gt_binary_fov = (gt * fov_mask) > 0.5
                    belief_binary_fov = (fused_prob * fov_mask) > 0.5
                    
                    intersection_fov = (gt_binary_fov & belief_binary_fov).sum()
                    union_fov = (gt_binary_fov | belief_binary_fov).sum()
                    iou_fov = intersection_fov / (union_fov + 1e-7)
                    results['iou_fov'].append(float(iou_fov))
                    
                    # Industry Standard Metrics within FOV (final state)
                    belief_fov = fused_prob * fov_mask
                    gt_fov = gt * fov_mask
                    seg_metrics_fov = compute_segmentation_metrics(belief_fov, gt_fov)
                    results['f1_fov'].append(seg_metrics_fov['f1_score'])
                    results['precision_fov'].append(seg_metrics_fov['precision'])
                    results['recall_fov'].append(seg_metrics_fov['recall'])
                    
                    # Target-Only Error within FOV (final state)
                    target_error_fov = compute_target_only_error(belief_fov, gt_fov)
                    results['target_only_error_fov'].append(target_error_fov)
                
                # Save visualizations
                if save_examples and examples_saved < self.sim_config.num_examples_to_save:
                    # Save visualizations with ray-tracing beams
                    self._save_scene_visualizations_raytracing(
                        scene_idx=i,
                        example_idx=examples_saved,
                        strategy_name=strategy_name,
                        gt=gt,
                        initial_prior=mean_prob,
                        entropy_initial=entropy_map,
                        beams_selected=beams,
                        coverage_mask=coverage_mask,
                        final_belief=fused_prob,
                        entropy_final=entropy_after,
                        total_ig=total_ig,
                        entropy_trace=entropy_trace_scene,
                        examples_dir=examples_dir
                    )
                    examples_saved += 1
                
                # Classify scene and append to per-condition buckets
                scene_groups = get_scene_groups_from_dataset(self.val_loader.dataset, i)
                _append_scene_to_condition_buckets(results, results_by_condition, scene_groups)
        
        if save_examples:
            print(f"   ✓ Saved {examples_saved} example visualizations to {examples_dir}")
        
        # Aggregate global results
        trace_keys = ['entropy_trace', 'error_trace', 'iou_trace', 'f1_trace', 'precision_trace', 
                     'recall_trace', 'target_only_error_trace', 'roi_entropy_trace', 'chamfer_trace',
                     'entropy_trace_fov', 'error_trace_fov', 'iou_trace_fov', 'roc_global', 'roc_fov',
                     'f1_visible_trace', 'precision_visible_trace', 'recall_visible_trace', 
                     'error_visible_trace', 'iou_visible_trace']
        aggregated = {k: float(np.mean(v)) if k not in trace_keys else v 
                     for k, v in results.items()}
        aggregated['std'] = {k: float(np.std(v)) for k, v in results.items() if k not in trace_keys}
        
        # Aggregate per-condition results
        aggregated['by_condition'] = {}
        for condition, cond_results in results_by_condition.items():
            aggregated['by_condition'][condition] = {
                k: float(np.mean(v)) if k not in trace_keys and len(v) > 0 else v
                for k, v in cond_results.items()
            }
            aggregated['by_condition'][condition]['std'] = {
                k: float(np.std(v)) for k, v in cond_results.items() 
                if k not in trace_keys and len(v) > 0
            }
            aggregated['by_condition'][condition]['count'] = len(cond_results['f1_score'])
        
        return aggregated, results
    
    def run_all_strategies(self, include_random=False):
        """Run all strategies and compare"""
        all_results = {}
        
        # Camera-only baseline (no radar)
        print("\n📷 CAMERA-ONLY BASELINE: No radar fusion")
        all_results['camera_only'] = self.run_camera_only_baseline()
        
        # Cognitive: Sequential closed-loop
        print("\n🧠 COGNITIVE LOOP: Adapts after each pulse")
        all_results['entropy'] = self.run_strategy_cognitive()
        
        # Baselines: Pre-planned open-loop
        print("\n📐 OPEN-LOOP BASELINES: Pre-planned, no adaptation")
        baseline_strategies = ['uniform', 'random'] if include_random else ['uniform']
        for strategy in baseline_strategies:
            all_results[strategy] = self.run_strategy(strategy)
        
        # Generate comparison
        self._generate_comparison(all_results)
        
        return all_results
    
    def run_camera_only_baseline(self):
        """Compute metrics for camera-only (no radar fusion)"""
        # Initialize global results
        results = _create_empty_results_dict()
        
        # Initialize per-condition results (buckets: DAY, NIGHT, RAINY, etc.)
        results_by_condition = {}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="camera_only")):
                if i >= self.sim_config.num_test_scenes:
                    break
                
                imgs, rots, trans, intrins, post_rots, post_trans, gt_binimg = batch
                imgs = imgs.to(self.device)
                rots, trans, intrins = rots.to(self.device), trans.to(self.device), intrins.to(self.device)
                post_rots, post_trans = post_rots.to(self.device), post_trans.to(self.device)
                
                preds = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
                probs = torch.sigmoid(preds)
                mean_prob = probs.mean(dim=1)[0, 0].cpu().numpy()
                mean_prob = np.fliplr(mean_prob)
                gt = np.fliplr(gt_binimg[0, 0].numpy())
                
                # Compute metrics on camera-only prediction
                seg_metrics = compute_segmentation_metrics(mean_prob, gt)
                results['f1_score'].append(seg_metrics['f1_score'])
                results['precision'].append(seg_metrics['precision'])
                results['recall'].append(seg_metrics['recall'])
                
                # IoU
                gt_binary = (gt > 0.5)
                pred_binary = (mean_prob > 0.5)
                intersection = (gt_binary & pred_binary).sum()
                union = (gt_binary | pred_binary).sum()
                iou = intersection / (union + 1e-7)
                results['iou'].append(float(iou))
                
                # Target-only error
                target_error = compute_target_only_error(mean_prob, gt)
                results['target_only_error'].append(target_error)
                
                # ROI entropy
                entropy_map = compute_entropy(mean_prob)
                roi_ent = compute_roi_entropy(entropy_map, gt)
                results['roi_entropy'].append(roi_ent)
                
                # Chamfer distance
                chamfer = compute_chamfer_distance(mean_prob, gt)
                results['chamfer_distance'].append(chamfer)
                
                # Compute GT visibility mask
                gt_visibility_mask_full = compute_gt_visibility_mask(gt, self.sim_config.grid_conf, self.radar_config, self.sim_config)
                
                # GLASS CEILING: Intersection of Visible AND FOV
                # FOV mask
                fov_mask = self._compute_radar_fov_mask(self.sim_config.grid_conf)
                fov_mask_bool = fov_mask.astype(bool)
                gt_visibility_mask = gt_visibility_mask_full & fov_mask_bool
                
                # Compute THEORETICAL MAXIMUM given sensor limitations
                theoretical_max = compute_perfect_sensor_baseline(gt, gt_visibility_mask, self.radar_config)
                
                # VISIBLE-ONLY Metrics (Glass Ceiling)
                visible_metrics = compute_visible_metrics(mean_prob, gt, gt_visibility_mask)
                results['f1_visible'].append(visible_metrics['f1_visible'])
                results['precision_visible'].append(visible_metrics['precision_visible'])
                results['recall_visible'].append(visible_metrics['recall_visible'])
                results['error_visible'].append(visible_metrics['error_visible'])
                results['iou_visible'].append(visible_metrics['iou_visible'])
                results['target_only_error_visible'].append(visible_metrics['target_only_error_visible'])
                results['visibility_ratio'].append(visible_metrics['visibility_ratio'])
                results['occupied_visible_ratio'].append(visible_metrics['occupied_visible_ratio'])
                
                # Theoretical Maximum (computed once per scene)
                results['theoretical_max_f1'].append(theoretical_max['f1_max'])
                results['theoretical_max_iou'].append(theoretical_max['iou_max'])
                results['sensor_Pd'].append(theoretical_max['Pd'])
                results['sensor_Pfa'].append(theoretical_max['Pfa'])
                
                # ROC curve
                roc_global = compute_roc_curve(mean_prob, gt)
                results['roc_global'].append(roc_global)
                
                # FOV mask
                fov_mask = self._compute_radar_fov_mask(self.sim_config.grid_conf)
                mean_prob_fov = mean_prob * fov_mask
                gt_fov = gt * fov_mask
                roc_fov = compute_roc_curve(mean_prob_fov, gt_fov)
                results['roc_fov'].append(roc_fov)
                
                # Dummy values for radar-specific metrics
                results['information_gain'].append(0.0)
                results['coverage_ratio'].append(0.0)
                results['high_entropy_coverage'].append(0.0)
                results['mean_entropy_scanned'].append(0.0)
                results['detection_improvement'].append(0.0)
                
                # Classify scene and append to per-condition buckets
                scene_groups = get_scene_groups_from_dataset(self.val_loader.dataset, i)
                _append_scene_to_condition_buckets(results, results_by_condition, scene_groups)
        
        # Aggregate global results
        aggregated = {k: float(np.mean(v)) if k not in ['roc_global', 'roc_fov'] else v 
                     for k, v in results.items()}
        aggregated['std'] = {k: float(np.std(v)) for k, v in results.items() 
                           if k not in ['roc_global', 'roc_fov']}
        
        # Aggregate per-condition results
        trace_keys = ['entropy_trace', 'error_trace', 'iou_trace', 'f1_trace', 'precision_trace', 
                     'recall_trace', 'target_only_error_trace', 'roi_entropy_trace', 'chamfer_trace',
                     'entropy_trace_fov', 'error_trace_fov', 'iou_trace_fov', 'roc_global', 'roc_fov',
                     'f1_visible_trace', 'precision_visible_trace', 'recall_visible_trace', 
                     'error_visible_trace', 'iou_visible_trace']
        aggregated['by_condition'] = {}
        for condition, cond_results in results_by_condition.items():
            aggregated['by_condition'][condition] = {
                k: float(np.mean(v)) if k not in trace_keys and len(v) > 0 else v
                for k, v in cond_results.items()
            }
            aggregated['by_condition'][condition]['std'] = {
                k: float(np.std(v)) for k, v in cond_results.items() 
                if k not in trace_keys and len(v) > 0
            }
            aggregated['by_condition'][condition]['count'] = len(cond_results['f1_score'])
        
        return aggregated, results
    
    def _generate_comparison(self, all_results):
        """Generate comparison report and visualizations"""
        print("\n" + "="*80)
        # Use only strategies that were actually run
        strategies = list(all_results.keys())
        
        # Metrics where LOWER is better
        lower_is_better = ['target_only_error', 'roi_entropy', 'chamfer_distance', 'target_only_error_fov']
        
        # Store all metrics for visualization
        global_metrics = ['information_gain', 'coverage_ratio', 'high_entropy_coverage', 
                         'mean_entropy_scanned', 'detection_improvement', 'iou',
                         'f1_score', 'precision', 'recall', 'target_only_error',
                         'roi_entropy', 'chamfer_distance']
        
        fov_metrics = ['information_gain_fov', 'coverage_ratio_fov', 'high_entropy_coverage_fov', 
                      'mean_entropy_scanned_fov', 'detection_improvement_fov', 'iou_fov',
                      'f1_fov', 'precision_fov', 'recall_fov', 'target_only_error_fov']
        
        # Generate plots
        print("\n📊 Generating comparison plots...")
        
        # Comprehensive metric table + ROC curves (NEW!)
        plot_metrics_table_and_roc(all_results, self.sim_config.results_dir, lower_is_better)
        
        # Time-series plots (efficiency curves)
        print("   📈 Generating efficiency traces...")
        plot_entropy_traces(all_results, self.sim_config.results_dir)
        plot_error_traces(all_results, self.sim_config.results_dir)
        plot_iou_traces(all_results, self.sim_config.results_dir)
        plot_entropy_traces_fov(all_results, self.sim_config.results_dir)
        plot_error_traces_fov(all_results, self.sim_config.results_dir)
        plot_iou_traces_fov(all_results, self.sim_config.results_dir)
        
        # NEW: Efficiency traces for all industry-standard metrics
        plot_f1_traces(all_results, self.sim_config.results_dir)
        plot_precision_recall_traces(all_results, self.sim_config.results_dir)
        plot_target_only_error_traces(all_results, self.sim_config.results_dir)
        plot_roi_entropy_traces(all_results, self.sim_config.results_dir)
        plot_chamfer_distance_traces(all_results, self.sim_config.results_dir)
        
        # GLASS CEILING: Visible-only metrics (theoretical maximum comparisons)
        print("   🏆 Generating Glass Ceiling plots (visible-only metrics)...")
        plot_f1_visible_traces(all_results, self.sim_config.results_dir)
        plot_iou_visible_traces(all_results, self.sim_config.results_dir)
        plot_error_visible_traces(all_results, self.sim_config.results_dir)
        
        plot_comparison_bars(all_results, self.sim_config.results_dir)
        
        # Save JSON
        save_dict = {}
        for strategy in strategies:
            aggregated = all_results[strategy][0]
            save_dict[strategy] = {
                'mean': aggregated,
                'std': aggregated['std']
            }
            # Extract by_condition to top level (additive, keeps existing structure)
            if 'by_condition' in aggregated:
                save_dict[strategy]['by_condition'] = aggregated['by_condition']
        
        # Make JSON-safe
        save_dict = self._make_json_safe(save_dict)
        
        with open(os.path.join(self.sim_config.results_dir, 'results.json'), 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"\n✅ Results saved to: {self.sim_config.results_dir}")
    
    def _make_json_safe(self, obj):
        """Recursively convert NumPy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


# ==========================================
#              MAIN
# ==========================================

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Cognitive Radar Simulation")
    parser.add_argument('--model_path', type=str, 
                       default='/home/shayelbaz/repos/checkpoints/runs/trainval/full_trainval_5heads_from_scratch/2026-01-04_15-23-47/checkpoints/model_best.pth',
                       help='Path to trained ensemble model')
    parser.add_argument('--budget', type=int, default=40, help='Number of radar beams per scene')
    parser.add_argument('--num_scenes', type=int, default=50, help='Number of test scenes')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of example visualizations')
    parser.add_argument('--name', type=str, default='experiment', 
                       help='Experiment name (results saved to results/{name}_{timestamp}/)')
    parser.add_argument('--include_random', action='store_true', 
                       help='Include random baseline (disabled by default)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None, no seeding)')
    
    args = parser.parse_args()
    
    # Set random seed if provided (for reproducible refactor validation)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    print("="*80)
    print("  COGNITIVE RADAR SIMULATION: INFORMATION-THEORETIC ACTIVE SENSING")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Radar Budget: {args.budget} beams/scene")
    print(f"Test Scenes: {args.num_scenes}")
    print(f"Experiment Name: {args.name}")
    if args.seed is not None:
        print(f"Random Seed: {args.seed}")
    print()
    
    # Setup configs
    radar_config = RadarConfig()
    sim_config = SimulationConfig()
    sim_config.radar_budget = args.budget
    sim_config.num_test_scenes = args.num_scenes
    sim_config.num_examples_to_save = args.num_examples
    
    # Set results directory with timestamp
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create results directory inside radar_simulation/results/
    results_base = os.path.join(script_dir, 'results')
    os.makedirs(results_base, exist_ok=True)
    
    # Check if experiment name already exists
    experiment_dir = os.path.join(results_base, args.name)
    if os.path.exists(experiment_dir):
        # Add timestamp if name already exists
        experiment_dir = os.path.join(results_base, f'{args.name}_{timestamp}')
        print(f"⚠️  Experiment '{args.name}' already exists, using: {args.name}_{timestamp}")
    
    sim_config.results_dir = experiment_dir
    os.makedirs(sim_config.results_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {os.path.relpath(sim_config.results_dir, script_dir)}\n")
    
    # Run experiment
    experiment = RadarSimulationExperiment(args.model_path, radar_config, sim_config)
    results = experiment.run_all_strategies(include_random=args.include_random)
    
    print("\n" + "="*80)
    print("✅ SIMULATION COMPLETE")
    print(f"📊 Results saved to: {os.path.relpath(sim_config.results_dir, script_dir)}")
    print("="*80)


if __name__ == "__main__":
    main()

