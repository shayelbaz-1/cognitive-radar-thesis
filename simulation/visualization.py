"""
Visualization functions for radar simulation results

This module provides all plotting and visualization functions:
- Time-series plots (entropy/error reduction over pulses)
- Beam placement visualizations
- Prior update sequences
- Comprehensive scene comparisons
- Animated GIFs showing cognitive loop evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
from information_theory import compute_entropy


def plot_entropy_traces_fov(all_results, save_dir):
    """
    Plot FOV-ONLY entropy reduction over time
    
    Shows entropy reduction in the radar-scannable region (120° FOV, 4-50m).
    This demonstrates radar's DIRECT impact where it can actually operate.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive (Closed-Loop)', 
             'uniform': 'Uniform (Open-Loop)', 
             'random': 'Random (Open-Loop)'}
    
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        
        if 'entropy_trace_fov' in results_dict:
            traces = results_dict['entropy_trace_fov']
            
            if len(traces) > 0 and all(len(t) > 0 for t in traces):
                max_len = max(len(t) for t in traces)
                padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
                avg_trace = np.mean(padded, axis=0)
                std_trace = np.std(padded, axis=0)
                
                pulses = np.arange(len(avg_trace))
                
                if strategy == 'entropy':
                    ax.plot(pulses, avg_trace, 'o-', color=colors[strategy], 
                           linewidth=5, markersize=12, label=labels[strategy], 
                           markerfacecolor=colors[strategy], markeredgecolor='white', 
                           markeredgewidth=2, alpha=1.0, zorder=3)
                elif strategy == 'uniform':
                    ax.plot(pulses, avg_trace, 's--', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=2, dashes=(5, 3))
                else:
                    ax.plot(pulses, avg_trace, '^:', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=1)
                
                ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                               color=colors[strategy], alpha=0.15)
    
    ax.set_xlabel('Pulse Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('FOV Entropy (bits)', fontsize=14, fontweight='bold')
    ax.set_title('FOV-Only Uncertainty Reduction (120° FOV, 4-50m)\n' +
                'Radar\'s DIRECT Impact in Scannable Region',
                fontsize=15, fontweight='bold', color='darkred')
    
    legend = ax.legend(loc='best', fontsize=14, framealpha=1.0, edgecolor='black', 
                      shadow=True, fancybox=True, borderpad=1.2)
    legend.get_frame().set_linewidth(2)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_trace_comparison_fov.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ FOV entropy trace plot saved")


def plot_error_traces_fov(all_results, save_dir):
    """
    Plot FOV-ONLY detection error reduction over time
    
    Shows MAE reduction in the radar-scannable region only.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random'}
    
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        
        if 'error_trace_fov' in results_dict:
            traces = results_dict['error_trace_fov']
            
            if len(traces) > 0 and all(len(t) > 0 for t in traces):
                max_len = max(len(t) for t in traces)
                padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
                avg_trace = np.mean(padded, axis=0)
                std_trace = np.std(padded, axis=0)
                
                pulses = np.arange(len(avg_trace))
                
                if strategy == 'entropy':
                    ax.plot(pulses, avg_trace, 'o-', color=colors[strategy], 
                           linewidth=5, markersize=12, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white', 
                           markeredgewidth=2, alpha=1.0, zorder=3)
                elif strategy == 'uniform':
                    ax.plot(pulses, avg_trace, 's--', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=2, dashes=(5, 3))
                else:
                    ax.plot(pulses, avg_trace, '^:', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=1)
                
                ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                               color=colors[strategy], alpha=0.15)
    
    ax.set_xlabel('Pulse Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('FOV Detection Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title('FOV-Only Detection Error Reduction (120° FOV, 4-50m)\n' +
                'Lower is Better',
                fontsize=15, fontweight='bold', color='darkred')
    
    legend = ax.legend(loc='best', fontsize=14, framealpha=1.0, edgecolor='black', 
                      shadow=True, fancybox=True, borderpad=1.2)
    legend.get_frame().set_linewidth(2)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_trace_comparison_fov.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ FOV error trace plot saved")


def plot_entropy_traces(all_results, save_dir):
    """
    Plot entropy reduction over time (per pulse) for all strategies
    
    This is THE KEY PLOT showing cognitive advantage:
    - Steeper descent = faster uncertainty reduction
    - Cognitive adapts → targets high entropy → faster learning
    
    Args:
        all_results: Dict with keys 'entropy', 'uniform', 'random'
                    Each contains (aggregated_metrics, raw_results)
        save_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive (Closed-Loop)', 
             'uniform': 'Uniform (Open-Loop)', 
             'random': 'Random (Open-Loop)'}
    
    # Plot average entropy trace for each strategy
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        
        if 'entropy_trace' in results_dict:
            traces = results_dict['entropy_trace']
            
            if len(traces) > 0 and all(len(t) > 0 for t in traces):
                # Average across scenes
                max_len = max(len(t) for t in traces)
                padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
                avg_trace = np.mean(padded, axis=0)
                std_trace = np.std(padded, axis=0)
                
                pulses = np.arange(len(avg_trace))
                
                # Plot with distinct styling
                if strategy == 'entropy':
                    ax.plot(pulses, avg_trace, 'o-', color=colors[strategy], 
                           linewidth=5, markersize=12, label=labels[strategy], 
                           markerfacecolor=colors[strategy], markeredgecolor='white', 
                           markeredgewidth=2, alpha=1.0, zorder=3)
                elif strategy == 'uniform':
                    ax.plot(pulses, avg_trace, 's--', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=2, dashes=(5, 3))
                else:  # random
                    ax.plot(pulses, avg_trace, '^:', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=1)
                
                ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                               color=colors[strategy], alpha=0.15)
    
    ax.set_xlabel('Pulse Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Map Entropy (bits)', fontsize=14, fontweight='bold')
    ax.set_title('Uncertainty Reduction: Cognitive vs Open-Loop Baselines\n' +
                'KEY RESULT: Steeper slope = faster learning',
                fontsize=15, fontweight='bold')
    
    legend = ax.legend(loc='best', fontsize=14, framealpha=1.0, edgecolor='black', 
                      shadow=True, fancybox=True, borderpad=1.2, 
                      markerscale=1.5, labelspacing=1.0)
    legend.get_frame().set_linewidth(2)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_trace_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Entropy trace plot saved")


def plot_error_traces(all_results, save_dir):
    """
    Plot detection error (MAE) reduction over time for all strategies
    
    Shows how predictions get closer to ground truth after each pulse.
    Lower error = better accuracy.
    
    Args:
        all_results: Dict with strategy results
        save_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive (Closed-Loop)', 
             'uniform': 'Uniform (Open-Loop)', 
             'random': 'Random (Open-Loop)'}
    
    # Calculate error reduction metrics
    error_reductions = {}
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        if 'error_trace' in results_dict:
            traces = results_dict['error_trace']
            if len(traces) > 0:
                avg_start = np.mean([t[0] for t in traces])
                avg_end = np.mean([t[-1] for t in traces])
                reduction = avg_start - avg_end
                reduction_pct = (reduction / avg_start) * 100
                error_reductions[strategy] = (reduction, reduction_pct)
    
    # Plot traces
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        
        if 'error_trace' in results_dict:
            traces = results_dict['error_trace']
            
            if len(traces) > 0 and all(len(t) > 0 for t in traces):
                max_len = max(len(t) for t in traces)
                padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
                avg_trace = np.mean(padded, axis=0)
                std_trace = np.std(padded, axis=0)
                
                pulses = np.arange(len(avg_trace))
                
                if strategy == 'entropy':
                    ax.plot(pulses, avg_trace, 'o-', color=colors[strategy], 
                           linewidth=5, markersize=12, label=labels[strategy], 
                           markerfacecolor=colors[strategy], markeredgecolor='white', 
                           markeredgewidth=2, alpha=1.0, zorder=3)
                elif strategy == 'uniform':
                    ax.plot(pulses, avg_trace, 's--', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=2, dashes=(5, 3))
                else:  # random
                    ax.plot(pulses, avg_trace, '^:', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=1)
                
                ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                               color=colors[strategy], alpha=0.15)
    
    # Build title with metrics
    title_text = 'Detection Accuracy: Getting Closer to Ground Truth\n'
    if error_reductions:
        cog_red, cog_pct = error_reductions.get('entropy', (0, 0))
        uni_red, uni_pct = error_reductions.get('uniform', (0, 0))
        ran_red, ran_pct = error_reductions.get('random', (0, 0))
        title_text += f'Error Reduction: Cognitive={cog_red:.4f} ({cog_pct:.1f}%), Uniform={uni_red:.4f} ({uni_pct:.1f}%), Random={ran_red:.4f} ({ran_pct:.1f}%)'
    
    ax.set_xlabel('Pulse Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Detection Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    legend = ax.legend(loc='upper right', fontsize=14, framealpha=1.0, edgecolor='black', 
                      shadow=True, fancybox=True, borderpad=1.2, 
                      markerscale=1.5, labelspacing=1.0)
    legend.get_frame().set_linewidth(2)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_trace_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Error trace plot saved")


def plot_comparison_bars(all_results, save_dir):
    """
    Bar chart comparison: Global vs FOV metrics side-by-side
    
    Left column: Global metrics (entire BEV)
    Right column: FOV metrics (radar scannable region)
    
    Args:
        all_results: Dict with strategy results
        save_dir: Directory to save plot
    """
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies_keys = [k for k in list(all_results.keys()) if k != 'camera_only']
    strategy_name_map = {'entropy': 'Entropy', 'uniform': 'Uniform', 'random': 'Random'}
    strategies_display = [strategy_name_map[k] for k in strategies_keys]
    color_map = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    colors = [color_map[k] for k in strategies_keys]
    
    # Create figure with 5 rows, 2 columns (global vs FOV)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Cognitive Radar: Global vs FOV Metrics Comparison\n' +
                 'Left: Entire BEV Map  |  Right: Radar FOV (120°, 4-50m)', 
                 fontsize=16, fontweight='bold')
    
    metrics_pairs = [
        ('information_gain', 'information_gain_fov', 'Information Gain (bits)'),
        ('high_entropy_coverage', 'high_entropy_coverage_fov', 'High-Entropy Coverage'),
        ('detection_improvement', 'detection_improvement_fov', 'Detection Improvement (MAE)'),
    ]
    
    for row_idx, (global_metric, fov_metric, label) in enumerate(metrics_pairs):
        # LEFT: Global metric
        ax_global = axes[row_idx, 0]
        means_global = [all_results[key][0][global_metric] for key in strategies_keys]
        stds_global = [all_results[key][0]['std'][global_metric] if global_metric in all_results[key][0]['std'] else 0 
                      for key in strategies_keys]
        
        bars_global = ax_global.bar(strategies_display, means_global, yerr=stds_global, 
                                    color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
        
        winner_idx_global = np.argmax(means_global)
        bars_global[winner_idx_global].set_edgecolor('gold')
        bars_global[winner_idx_global].set_linewidth(4)
        
        ax_global.set_ylabel(label, fontsize=11, fontweight='bold')
        ax_global.set_title(f'GLOBAL: {label}', fontsize=12, fontweight='bold')
        ax_global.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars_global:
            height = bar.get_height()
            ax_global.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # RIGHT: FOV metric
        ax_fov = axes[row_idx, 1]
        means_fov = [all_results[key][0][fov_metric] for key in strategies_keys]
        stds_fov = [all_results[key][0]['std'][fov_metric] if fov_metric in all_results[key][0]['std'] else 0 
                   for key in strategies_keys]
        
        bars_fov = ax_fov.bar(strategies_display, means_fov, yerr=stds_fov, 
                              color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
        
        winner_idx_fov = np.argmax(means_fov)
        bars_fov[winner_idx_fov].set_edgecolor('gold')
        bars_fov[winner_idx_fov].set_linewidth(4)
        
        ax_fov.set_ylabel(label, fontsize=11, fontweight='bold')
        ax_fov.set_title(f'FOV ONLY: {label}', fontsize=12, fontweight='bold', color='darkred')
        ax_fov.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars_fov:
            height = bar.get_height()
            ax_fov.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add advantage ratio annotation
        if means_fov[0] > 0 and means_fov[1] > 0:
            ratio = means_fov[0] / means_fov[1]
            ax_fov.text(0.5, 0.95, f'Cognitive Advantage: {ratio:.1f}×',
                       transform=ax_fov.transAxes, ha='center', va='top',
                       fontsize=10, fontweight='bold', color='green',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_global_vs_fov.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Global vs FOV comparison saved")


def save_simple_comparison(scene_idx, strategy_name, camera_prob, entropy_map, 
                            coverage_mask, fused_prob, entropy_after, grid_conf, save_path, gt=None):
    """
    Simple 2x2 before/after comparison WITH ground truth overlay
    
    Layout:
    [Entropy Before + Coverage] [Entropy After]
    [Camera Prediction + GT]     [Fused Prediction + GT]
    
    Args:
        scene_idx: Scene number
        strategy_name: Strategy name
        camera_prob: Initial camera prediction
        entropy_map: Initial entropy
        coverage_mask: Where radar scanned
        fused_prob: Final fused belief
        entropy_after: Final entropy
        grid_conf: BEV grid configuration
        save_path: Where to save
        gt: Ground truth occupancy map (optional, for overlay)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    xbound, ybound = grid_conf['xbound'], grid_conf['ybound']
    extent = [xbound[0], xbound[1], ybound[0], ybound[1]]
    
    # Top-left: Entropy Before with coverage
    if gt is not None:
        ax1.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.2, vmin=0, vmax=1)
    im1 = ax1.imshow(entropy_map, cmap='inferno', origin='lower', extent=extent, alpha=0.7)
    ax1.contourf(coverage_mask, levels=[0.3, 1.0], colors=['cyan'], 
                extent=extent, origin='lower', alpha=0.3)
    ax1.set_title('Entropy BEFORE Radar\n(Cyan = Scanned, Gray = GT)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m) - RIGHT →')
    ax1.set_ylabel('Y (m) - FORWARD ↑')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Entropy (bits)')
    ax1.plot(0, 0, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=1)
    ax1.set_aspect('equal', adjustable='box')
    
    # Top-right: Entropy After
    if gt is not None:
        ax2.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.2, vmin=0, vmax=1)
    im2 = ax2.imshow(entropy_after, cmap='inferno', origin='lower', extent=extent, alpha=0.7)
    ax2.set_title('Entropy AFTER Radar\n(Darker = More Certain)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m) - RIGHT →')
    ax2.set_ylabel('Y (m) - FORWARD ↑')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Entropy (bits)')
    ax2.plot(0, 0, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=1)
    ax2.set_aspect('equal', adjustable='box')
    
    # Bottom-left: Camera Prediction
    if gt is not None:
        ax3.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)
    im3 = ax3.imshow(camera_prob, cmap='jet', origin='lower', extent=extent, vmin=0, vmax=1, alpha=0.7)
    ax3.set_title('Camera Prediction\n(Before Radar, Gray=GT)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (m) - RIGHT →')
    ax3.set_ylabel('Y (m) - FORWARD ↑')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Occupancy Prob')
    ax3.plot(0, 0, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=1)
    ax3.set_aspect('equal', adjustable='box')
    
    # Bottom-right: Fused Prediction
    if gt is not None:
        ax4.imshow(gt, cmap='gray', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1)
    im4 = ax4.imshow(fused_prob, cmap='jet', origin='lower', extent=extent, vmin=0, vmax=1, alpha=0.7)
    ax4.set_title('Fused Prediction\n(Camera + Radar, Gray=GT)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X (m) - RIGHT →')
    ax4.set_ylabel('Y (m) - FORWARD ↑')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Occupancy Prob')
    ax4.plot(0, 0, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=1)
    ax4.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f'{strategy_name.upper()} Strategy - Scene {scene_idx+1}\n' +
                'Radar reduces uncertainty | Gray overlay = Ground Truth',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_iou_traces(all_results, save_dir):
    """
    Plot IoU improvement over time (per pulse) for all strategies
    
    IoU (Intersection over Union) measures object blob detection quality:
    - Higher = better object shape capture
    - Shows how well radar finds and maps car-sized blobs
    
    Args:
        all_results: Dict with keys 'entropy', 'uniform', 'random'
        save_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive (Closed-Loop)', 
             'uniform': 'Uniform (Open-Loop)', 
             'random': 'Random (Open-Loop)'}
    
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        
        if 'iou_trace' in results_dict:
            traces = results_dict['iou_trace']
            
            if len(traces) > 0 and all(len(t) > 0 for t in traces):
                max_len = max(len(t) for t in traces)
                padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
                avg_trace = np.mean(padded, axis=0)
                std_trace = np.std(padded, axis=0)
                
                pulses = np.arange(len(avg_trace))
                
                if strategy == 'entropy':
                    ax.plot(pulses, avg_trace, 'o-', color=colors[strategy], 
                           linewidth=5, markersize=12, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white', 
                           markeredgewidth=2, alpha=1.0, zorder=3)
                elif strategy == 'uniform':
                    ax.plot(pulses, avg_trace, 's--', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=2, dashes=(5, 3))
                else:
                    ax.plot(pulses, avg_trace, '^:', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=1)
                
                ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                               color=colors[strategy], alpha=0.15)
    
    ax.set_xlabel('Pulse Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    ax.set_title('Object Blob Detection Quality Over Time\n' +
                'IoU: How Well Radar Captures Car Shapes (Higher is Better)',
                fontsize=15, fontweight='bold', color='darkgreen')
    
    legend = ax.legend(loc='best', fontsize=14, framealpha=1.0, edgecolor='black', 
                      shadow=True, fancybox=True, borderpad=1.2)
    legend.get_frame().set_linewidth(2)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_ylim([0, 1.0])  # IoU is between 0 and 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_trace_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ IoU trace plot saved")


def plot_iou_traces_fov(all_results, save_dir):
    """
    Plot FOV-ONLY IoU improvement over time
    
    Shows object blob detection quality within the radar-scannable region (120° FOV, 4-50m).
    This demonstrates radar's DIRECT impact where it can actually operate.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive (Closed-Loop)', 
             'uniform': 'Uniform (Open-Loop)', 
             'random': 'Random (Open-Loop)'}
    
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        
        if 'iou_trace_fov' in results_dict:
            traces = results_dict['iou_trace_fov']
            
            if len(traces) > 0 and all(len(t) > 0 for t in traces):
                max_len = max(len(t) for t in traces)
                padded = [t + [t[-1]] * (max_len - len(t)) for t in traces]
                avg_trace = np.mean(padded, axis=0)
                std_trace = np.std(padded, axis=0)
                
                pulses = np.arange(len(avg_trace))
                
                if strategy == 'entropy':
                    ax.plot(pulses, avg_trace, 'o-', color=colors[strategy], 
                           linewidth=5, markersize=12, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white', 
                           markeredgewidth=2, alpha=1.0, zorder=3)
                elif strategy == 'uniform':
                    ax.plot(pulses, avg_trace, 's--', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=2, dashes=(5, 3))
                else:
                    ax.plot(pulses, avg_trace, '^:', color=colors[strategy], 
                           linewidth=5, markersize=11, label=labels[strategy],
                           markerfacecolor=colors[strategy], markeredgecolor='white',
                           markeredgewidth=2, alpha=1.0, zorder=1)
                
                ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                               color=colors[strategy], alpha=0.15)
    
    ax.set_xlabel('Pulse Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('FOV IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    ax.set_title('FOV-Only Object Blob Detection Quality (120° FOV, 4-50m)\n' +
                'Higher is Better',
                fontsize=15, fontweight='bold', color='darkgreen')
    
    legend = ax.legend(loc='best', fontsize=14, framealpha=1.0, edgecolor='black', 
                      shadow=True, fancybox=True, borderpad=1.2)
    legend.get_frame().set_linewidth(2)
    
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_ylim([0, 1.0])  # IoU is between 0 and 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_trace_comparison_fov.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ FOV IoU trace plot saved")


def plot_comprehensive_metric_summary(all_results, save_dir, lower_is_better_metrics):
    """
    Create comprehensive metric summary visualization
    
    Shows all key metrics in bar charts with clear visual hierarchy
    """
    strategies = list(all_results.keys())
    strategy_labels = {'entropy': 'Cognitive\n(Closed-Loop)', 'uniform': 'Uniform\n(Baseline)', 'random': 'Random\n(Baseline)'}
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    
    # Industry Standard Metrics
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle('INDUSTRY STANDARD METRICS (Safety-Critical)', fontsize=18, fontweight='bold', y=0.995)
    
    metrics_row1 = [
        ('f1_score', 'F1-Score', 'Harmonic Mean of Precision & Recall'),
        ('precision', 'Precision', 'How many detected cars are real?'),
        ('recall', 'Recall', 'How many real cars did we find?')
    ]
    
    metrics_row2 = [
        ('iou', 'IoU', 'Object Blob Detection Quality'),
        ('target_only_error', 'Target-Only Error', 'MAE on Occupied Cells (Lower=Better)'),
        ('chamfer_distance', 'Chamfer Distance', 'Geometric Closeness (pixels, Lower=Better)')
    ]
    
    for idx, (metric, title, subtitle) in enumerate(metrics_row1):
        ax = axes1[0, idx]
        values = [all_results[s][0][metric] for s in strategies]
        bars = ax.bar([strategy_labels[s] for s in strategies], values, 
                     color=[colors[s] for s in strategies], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Determine winner
        if metric in lower_is_better_metrics:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for idx, (metric, title, subtitle) in enumerate(metrics_row2):
        ax = axes1[1, idx]
        values = [all_results[s][0][metric] for s in strategies]
        bars = ax.bar([strategy_labels[s] for s in strategies], values,
                     color=[colors[s] for s in strategies], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Determine winner
        if metric in lower_is_better_metrics:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary_industry_standard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Industry standard metrics summary saved")
    
    # Information-Theoretic Metrics
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('INFORMATION-THEORETIC & ACTIVE SENSING METRICS', fontsize=18, fontweight='bold', y=0.995)
    
    info_metrics = [
        ('information_gain', 'Information Gain', 'Total Uncertainty Reduction (bits)'),
        ('roi_entropy', 'ROI Entropy', 'Uncertainty in Target Regions (Lower=Better)'),
        ('mean_entropy_scanned', 'Mean Entropy Scanned', 'Quality of Beam Placement'),
        ('high_entropy_coverage', 'High-Entropy Coverage', 'Fraction of Uncertain Regions Scanned'),
        ('coverage_ratio', 'Coverage Ratio', 'Fraction of Map Scanned'),
        ('detection_improvement', 'Detection Improvement', 'MAE Reduction from Camera-Only')
    ]
    
    for idx, (metric, title, subtitle) in enumerate(info_metrics):
        ax = axes2.flatten()[idx]
        values = [all_results[s][0][metric] for s in strategies]
        bars = ax.bar([strategy_labels[s] for s in strategies], values,
                     color=[colors[s] for s in strategies], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Determine winner
        if metric in lower_is_better_metrics:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary_information_theoretic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Information-theoretic metrics summary saved")
    
    # FOV-Specific Metrics (Most Important for Safety!)
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
    fig3.suptitle('FOV METRICS (Radar Scannable Region: 120° FOV, 4-50m) - SAFETY CRITICAL', 
                  fontsize=18, fontweight='bold', y=0.995, color='darkred')
    
    fov_metrics = [
        ('f1_fov', 'F1-Score (FOV)', 'Object Detection in Radar Range'),
        ('precision_fov', 'Precision (FOV)', 'Accuracy of Detections'),
        ('recall_fov', 'Recall (FOV)', 'Completeness of Detections'),
        ('iou_fov', 'IoU (FOV)', 'Object Shape Quality'),
        ('target_only_error_fov', 'Target Error (FOV)', 'MAE on Targets (Lower=Better)'),
        ('information_gain_fov', 'Information Gain (FOV)', 'Uncertainty Reduction (bits)')
    ]
    
    for idx, (metric, title, subtitle) in enumerate(fov_metrics):
        ax = axes3.flatten()[idx]
        values = [all_results[s][0][metric] for s in strategies]
        bars = ax.bar([strategy_labels[s] for s in strategies], values,
                     color=[colors[s] for s in strategies], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Determine winner
        if metric in lower_is_better_metrics:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary_fov.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ FOV metrics summary saved")
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_metrics_table_and_roc(all_results, save_dir, lower_is_better_metrics):
    """
    Create clean metric table + ROC curves
    
    Shows all metrics in professional tables with ROC curves for safety analysis
    """
    strategies = list(all_results.keys())
    
    # Create figure with table + ROC curves
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # === TOP: INDUSTRY STANDARD METRICS TABLE ===
    ax_table1 = fig.add_subplot(gs[0, :])
    ax_table1.axis('off')
    
    table_data = []
    headers = ['Metric', 'Camera Only', 'Cognitive', 'Uniform', 'Winner', 'Advantage']
    
    industry_metrics = [
        ('f1_score', 'F1-Score', False),
        ('precision', 'Precision', False),
        ('recall', 'Recall', False),
        ('iou', 'IoU', False),
        ('target_only_error', 'Target-Only Error', True),
        ('chamfer_distance', 'Chamfer Distance (px)', True),
    ]
    
    for metric_key, metric_name, lower_better in industry_metrics:
        row = [metric_name]
        
        # Camera-only value
        camera_val = all_results.get('camera_only', ({}, {}))[0].get(metric_key, 0.0)
        row.append(f'{camera_val:.4f}')
        
        # Strategy values
        values = []
        for s in strategies:
            if s != 'camera_only':
                val = all_results[s][0].get(metric_key, 0.0)
                values.append(val)
                row.append(f'{val:.4f}')
        
        # Winner
        if lower_better:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        winner_name = [s for s in strategies if s != 'camera_only'][winner_idx]
        row.append(winner_name.upper())
        
        # Cognitive advantage
        if 'entropy' in strategies:
            cog_val = all_results['entropy'][0].get(metric_key, 0.0)
            if lower_better:
                advantage = ((camera_val - cog_val) / (camera_val + 1e-7)) * 100
            else:
                advantage = ((cog_val - camera_val) / (camera_val + 1e-7)) * 100
            row.append(f'{advantage:+.1f}%')
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    table1 = ax_table1.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table1[(0, i)].set_facecolor('#2C3E50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(len(table_data)):
        for j in range(len(headers)):
            if j == 0:  # Metric name column
                table1[(i+1, j)].set_facecolor('#ECF0F1')
                table1[(i+1, j)].set_text_props(weight='bold')
            elif j == 4:  # Winner column
                table1[(i+1, j)].set_facecolor('#F39C12')
                table1[(i+1, j)].set_text_props(weight='bold')
            elif j == 5:  # Advantage column
                val_str = table_data[i][j]
                if '+' in val_str:
                    table1[(i+1, j)].set_facecolor('#2ECC71')
                    table1[(i+1, j)].set_text_props(weight='bold', color='white')
    
    ax_table1.set_title('INDUSTRY STANDARD METRICS (Safety-Critical)', 
                       fontsize=16, fontweight='bold', pad=20)
    
    # === MIDDLE: INFORMATION-THEORETIC METRICS TABLE ===
    ax_table2 = fig.add_subplot(gs[1, :])
    ax_table2.axis('off')
    
    table_data2 = []
    
    info_metrics = [
        ('information_gain', 'Information Gain (bits)', False),
        ('roi_entropy', 'ROI Entropy', True),
        ('mean_entropy_scanned', 'Mean Entropy Scanned', False),
        ('high_entropy_coverage', 'High-Entropy Coverage', False),
        ('coverage_ratio', 'Coverage Ratio', False),
        ('detection_improvement', 'Detection Improvement', False),
    ]
    
    for metric_key, metric_name, lower_better in info_metrics:
        row = [metric_name]
        
        camera_val = all_results.get('camera_only', ({}, {}))[0].get(metric_key, 0.0)
        row.append(f'{camera_val:.4f}')
        
        values = []
        for s in strategies:
            if s != 'camera_only':
                val = all_results[s][0].get(metric_key, 0.0)
                values.append(val)
                row.append(f'{val:.4f}')
        
        if lower_better:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        winner_name = [s for s in strategies if s != 'camera_only'][winner_idx]
        row.append(winner_name.upper())
        
        if 'entropy' in strategies:
            cog_val = all_results['entropy'][0].get(metric_key, 0.0)
            if lower_better:
                advantage = ((camera_val - cog_val) / (camera_val + 1e-7)) * 100
            else:
                advantage = ((cog_val - camera_val) / (camera_val + 1e-7)) * 100
            row.append(f'{advantage:+.1f}%')
        else:
            row.append('N/A')
        
        table_data2.append(row)
    
    table2 = ax_table2.table(cellText=table_data2, colLabels=headers,
                            cellLoc='center', loc='center',
                            colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 2.5)
    
    for i in range(len(headers)):
        table2[(0, i)].set_facecolor('#2C3E50')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(table_data2)):
        for j in range(len(headers)):
            if j == 0:
                table2[(i+1, j)].set_facecolor('#ECF0F1')
                table2[(i+1, j)].set_text_props(weight='bold')
            elif j == 4:
                table2[(i+1, j)].set_facecolor('#F39C12')
                table2[(i+1, j)].set_text_props(weight='bold')
            elif j == 5:
                val_str = table_data2[i][j]
                if '+' in val_str:
                    table2[(i+1, j)].set_facecolor('#2ECC71')
                    table2[(i+1, j)].set_text_props(weight='bold', color='white')
    
    ax_table2.set_title('INFORMATION-THEORETIC & ACTIVE SENSING METRICS', 
                       fontsize=16, fontweight='bold', pad=20)
    
    # === THIRD: GLASS CEILING METRICS (Visible-Only Performance) ===
    ax_table3 = fig.add_subplot(gs[2, :])
    ax_table3.axis('off')
    
    table_data3 = []
    
    glass_ceiling_metrics = [
        ('f1_visible', 'F1-Score (Visible Only)', False),
        ('iou_visible', 'IoU (Visible Only)', False),
        ('error_visible', 'MAE (Visible Only)', True),
        ('target_only_error_visible', 'Target Error (Visible)', True),
        ('visibility_ratio', 'Map Visibility %', False),
        ('occupied_visible_ratio', 'Target Visibility %', False),
    ]
    
    for metric_key, metric_name, lower_better in glass_ceiling_metrics:
        row = [metric_name]
        
        camera_val = all_results.get('camera_only', ({}, {}))[0].get(metric_key, 0.0)
        # Convert ratios to percentages for display
        if 'ratio' in metric_key:
            row.append(f'{camera_val*100:.1f}%')
        else:
            row.append(f'{camera_val:.4f}')
        
        values = []
        for s in strategies:
            if s != 'camera_only':
                val = all_results[s][0].get(metric_key, 0.0)
                values.append(val)
                if 'ratio' in metric_key:
                    row.append(f'{val*100:.1f}%')
                else:
                    row.append(f'{val:.4f}')
        
        if lower_better:
            winner_idx = np.argmin(values)
        else:
            winner_idx = np.argmax(values)
        winner_name = [s for s in strategies if s != 'camera_only'][winner_idx]
        row.append(winner_name.upper())
        
        if 'entropy' in strategies:
            cog_val = all_results['entropy'][0].get(metric_key, 0.0)
            if lower_better:
                advantage = ((camera_val - cog_val) / (camera_val + 1e-7)) * 100
            else:
                advantage = ((cog_val - camera_val) / (camera_val + 1e-7)) * 100
            row.append(f'{advantage:+.1f}%')
        else:
            row.append('N/A')
        
        table_data3.append(row)
    
    table3 = ax_table3.table(cellText=table_data3, colLabels=headers,
                            cellLoc='center', loc='center',
                            colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
    table3.auto_set_font_size(False)
    table3.set_fontsize(11)
    table3.scale(1, 2.5)
    
    for i in range(len(headers)):
        table3[(0, i)].set_facecolor('#8E44AD')  # Purple for Glass Ceiling
        table3[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(table_data3)):
        for j in range(len(headers)):
            if j == 0:
                table3[(i+1, j)].set_facecolor('#F4ECF7')  # Light purple
                table3[(i+1, j)].set_text_props(weight='bold')
            elif j == 4:
                table3[(i+1, j)].set_facecolor('#F39C12')
                table3[(i+1, j)].set_text_props(weight='bold')
            elif j == 5:
                val_str = table_data3[i][j]
                if '+' in val_str:
                    table3[(i+1, j)].set_facecolor('#2ECC71')
                    table3[(i+1, j)].set_text_props(weight='bold', color='white')
    
    ax_table3.set_title('🏆 GLASS CEILING METRICS (Visible + In FOV)\n' +
                       'Performance vs. Theoretical Maximum on Radar\'s Achievable Region', 
                       fontsize=16, fontweight='bold', pad=20, color='#8E44AD')
    
    # === BOTTOM: ROC CURVES ===
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c', 'camera_only': '#95a5a6'}
    labels = {'entropy': 'Cognitive (Closed-Loop)', 'uniform': 'Uniform (Baseline)', 
             'random': 'Random (Baseline)', 'camera_only': 'Camera Only'}
    
    # ROC Curve - Global
    ax_roc1 = fig.add_subplot(gs[3, 0])
    for strategy in strategies:
        if 'roc_global' in all_results[strategy][1]:
            roc_data = all_results[strategy][1]['roc_global'][0]  # First scene
            ax_roc1.plot(roc_data['fpr'], roc_data['tpr'], 
                        color=colors.get(strategy, '#333333'), 
                        linewidth=3, label=f"{labels.get(strategy, strategy)} (AUC={roc_data['auc']:.3f})",
                        alpha=0.9)
    
    ax_roc1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3, label='Random Classifier')
    ax_roc1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc1.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax_roc1.set_title('ROC Curve - Global BEV', fontsize=14, fontweight='bold')
    ax_roc1.legend(loc='lower right', fontsize=10)
    ax_roc1.grid(alpha=0.3)
    ax_roc1.set_xlim([0, 1])
    ax_roc1.set_ylim([0, 1])
    
    # ROC Curve - FOV
    ax_roc2 = fig.add_subplot(gs[3, 1])
    for strategy in strategies:
        if 'roc_fov' in all_results[strategy][1]:
            roc_data = all_results[strategy][1]['roc_fov'][0]  # First scene
            ax_roc2.plot(roc_data['fpr'], roc_data['tpr'], 
                        color=colors.get(strategy, '#333333'), 
                        linewidth=3, label=f"{labels.get(strategy, strategy)} (AUC={roc_data['auc']:.3f})",
                        alpha=0.9)
    
    ax_roc2.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3, label='Random Classifier')
    ax_roc2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc2.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax_roc2.set_title('ROC Curve - FOV (120° FOV, 1-50m)', fontsize=14, fontweight='bold', color='darkred')
    ax_roc2.legend(loc='lower right', fontsize=10)
    ax_roc2.grid(alpha=0.3)
    ax_roc2.set_xlim([0, 1])
    ax_roc2.set_ylim([0, 1])
    
    # Precision-Recall Curve - FOV
    ax_pr = fig.add_subplot(gs[3, 2])
    for strategy in strategies:
        if 'roc_fov' in all_results[strategy][1]:
            roc_data = all_results[strategy][1]['roc_fov'][0]
            # Compute precision from ROC data
            tpr = roc_data['tpr']
            fpr = roc_data['fpr']
            # Precision = TP / (TP + FP)
            # Need to estimate from TPR and FPR
            ax_pr.plot(tpr, tpr / (tpr + fpr + 1e-7),  # Approximate precision
                      color=colors.get(strategy, '#333333'), 
                      linewidth=3, label=labels.get(strategy, strategy),
                      alpha=0.9)
    
    ax_pr.set_xlabel('Recall (TPR)', fontsize=12, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_pr.set_title('Precision-Recall Curve - FOV', fontsize=14, fontweight='bold', color='darkred')
    ax_pr.legend(loc='lower left', fontsize=10)
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1])
    
    plt.savefig(os.path.join(save_dir, 'metrics_complete_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Complete metrics summary with ROC curves saved")


def plot_f1_traces(all_results, save_dir):
    """
    Plot F1-Score improvement over time - EFFICIENCY PROOF
    
    Shows how fast each strategy improves detection quality.
    Goal: Cognitive reaches F1=0.8 in 5 beams, Uniform takes 15+ beams.
    """
    # Use only strategies that were actually run (exclude camera_only baseline)
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive (Adaptive)', 'uniform': 'Uniform (Baseline)', 'random': 'Random (Baseline)'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        f1_trace = results_dict['f1_trace']
        
        # Average across all scenes
        max_len = max(len(trace) for trace in f1_trace)
        padded_traces = []
        for trace in f1_trace:
            if len(trace) < max_len:
                # Pad with last value
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        
        pulses = np.arange(len(avg_trace))
        
        ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
               label=labels[strategy], marker='o', markersize=5, alpha=0.9)
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.2)
    
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target F1=0.8')
    ax.set_xlabel('Beam Number (Radar Pulses)', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
    ax.set_title('F1-Score Efficiency: How Fast Does Detection Improve?\n' +
                'Cognitive reaches high F1 with fewer beams (Resource Efficiency)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_trace_efficiency.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ F1-Score efficiency trace saved")


def plot_precision_recall_traces(all_results, save_dir):
    """
    Plot Precision and Recall improvement over time
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Precision
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        precision_trace = results_dict['precision_trace']
        
        max_len = max(len(trace) for trace in precision_trace)
        padded_traces = []
        for trace in precision_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        ax1.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
                label=labels[strategy], marker='o', markersize=5, alpha=0.9)
        ax1.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                         color=colors[strategy], alpha=0.2)
    
    ax1.set_xlabel('Beam Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision Over Time\n(How many detections are correct?)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Recall
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        recall_trace = results_dict['recall_trace']
        
        max_len = max(len(trace) for trace in recall_trace)
        padded_traces = []
        for trace in recall_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        ax2.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
                label=labels[strategy], marker='o', markersize=5, alpha=0.9)
        ax2.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                         color=colors[strategy], alpha=0.2)
    
    ax2.set_xlabel('Beam Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_title('Recall Over Time\n(How many real objects did we find?)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_recall_traces.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Precision/Recall traces saved")


def plot_target_only_error_traces(all_results, save_dir):
    """
    Plot Target-Only Error over time (LOWER IS BETTER)
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        error_trace = results_dict['target_only_error_trace']
        
        max_len = max(len(trace) for trace in error_trace)
        padded_traces = []
        for trace in error_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
               label=labels[strategy], marker='o', markersize=5, alpha=0.9)
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.2)
    
    ax.set_xlabel('Beam Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Target-Only Error (MAE on Occupied Cells)', fontsize=13, fontweight='bold')
    ax.set_title('Target Error Reduction Over Time ⬇ LOWER IS BETTER\n' +
                'How accurately do we predict occupied regions?', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # Invert so "better" is visually "up"
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'target_only_error_trace.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Target-Only Error trace saved")


def plot_roi_entropy_traces(all_results, save_dir):
    """
    Plot ROI Entropy over time (LOWER IS BETTER)
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        roi_entropy_trace = results_dict['roi_entropy_trace']
        
        max_len = max(len(trace) for trace in roi_entropy_trace)
        padded_traces = []
        for trace in roi_entropy_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
               label=labels[strategy], marker='o', markersize=5, alpha=0.9)
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.2)
    
    ax.set_xlabel('Beam Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('ROI Entropy (bits)', fontsize=13, fontweight='bold')
    ax.set_title('Target Region Uncertainty Reduction ⬇ LOWER IS BETTER\n' +
                'How uncertain are we about occupied regions?', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # Invert so "better" is visually "up"
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roi_entropy_trace.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ ROI Entropy trace saved")


def plot_chamfer_distance_traces(all_results, save_dir):
    """
    Plot Chamfer Distance over time (LOWER IS BETTER)
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        chamfer_trace = results_dict['chamfer_trace']
        
        max_len = max(len(trace) for trace in chamfer_trace)
        padded_traces = []
        for trace in chamfer_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
               label=labels[strategy], marker='o', markersize=5, alpha=0.9)
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.2)
    
    ax.set_xlabel('Beam Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Chamfer Distance (pixels)', fontsize=13, fontweight='bold')
    ax.set_title('Geometric Accuracy Improvement ⬇ LOWER IS BETTER\n' +
                'How close are predicted shapes to ground truth?', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # Invert so "better" is visually "up"
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'chamfer_distance_trace.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Chamfer Distance trace saved")


def plot_f1_visible_traces(all_results, save_dir):
    """
    Plot F1-Score on VISIBLE pixels only - THE GLASS CEILING TEST
    
    Shows true performance relative to what's physically achievable.
    Includes TWO theoretical lines:
    1. Perfect sensor (F1 = 1.0) - mathematically impossible with real radar
    2. Realistic maximum (accounting for Pd < 1.0, Pfa > 0) - achievable ceiling
    
    This proves: "We achieve X% of the REALISTIC theoretical maximum."
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c', 'camera_only': '#95a5a6'}
    labels = {'entropy': 'Cognitive (Adaptive)', 'uniform': 'Uniform (Baseline)', 
             'random': 'Random (Baseline)', 'camera_only': 'Camera Only'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot camera-only baseline if available
    if 'camera_only' in all_results:
        aggregated_cam, _ = all_results['camera_only']
        f1_cam = aggregated_cam['f1_visible']
        ax.axhline(y=f1_cam, color=colors['camera_only'], linestyle=':', 
                  linewidth=3, alpha=0.6, label=labels['camera_only'], zorder=1)
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        f1_trace = results_dict['f1_visible_trace']
        
        # Average across all scenes
        max_len = max(len(trace) for trace in f1_trace)
        padded_traces = []
        for trace in f1_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        if strategy == 'entropy':
            ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=4, 
                   label=labels[strategy], marker='o', markersize=8, alpha=0.95, zorder=3)
        else:
            ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
                   label=labels[strategy], marker='s', markersize=6, alpha=0.85, zorder=2,
                   linestyle='--')
        
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.15)
    
    # THEORETICAL MAXIMUMS
    # Perfect sensor (impossible)
    ax.axhline(y=1.0, color='lightgray', linestyle=':', linewidth=3, alpha=0.5, 
              label='Perfect Sensor (F1=1.0, impossible)', zorder=4)
    
    # Realistic maximum (accounting for sensor limitations)
    # Check if any strategy has the theoretical_max_f1 value
    realistic_max = None
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        if 'theoretical_max_f1' in aggregated:
            realistic_max = aggregated['theoretical_max_f1']
            break
    
    if realistic_max and realistic_max > 0:
        ax.axhline(y=realistic_max, color='gold', linestyle='-.', linewidth=4, alpha=0.9, 
                  label=f'🏆 Realistic Maximum (F1={realistic_max:.3f}, Pd={aggregated.get("sensor_Pd", 0.9):.2f})', 
                  zorder=5)
        ax.fill_between([0, max_len-1], [realistic_max*0.95, realistic_max*0.95], [realistic_max, realistic_max], 
                        color='gold', alpha=0.15, label='Achievable Zone')
    else:
        # Fallback to perfect sensor line
        ax.axhline(y=1.0, color='gold', linestyle='-.', linewidth=4, alpha=0.8, 
                  label='🏆 Theoretical Maximum (100%)', zorder=4)
        ax.fill_between([0, max_len-1], [0.95, 0.95], [1.0, 1.0], 
                        color='gold', alpha=0.1, label='Achievable Zone')
    
    # Compute average visibility ratio to show in title
    visibility_ratios = []
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        if 'visibility_ratio' in results_dict and len(results_dict['visibility_ratio']) > 0:
            visibility_ratios.append(np.mean(results_dict['visibility_ratio']))
    
    if visibility_ratios:
        avg_visibility = np.mean(visibility_ratios) * 100
        ax.set_title(f'F1-Score: Visible Targets in FOV (Glass Ceiling Test)\n' +
                    f'Average {avg_visibility:.1f}% of map = Visible AND in FOV (radar\'s achievable region)\n' +
                    f'Gold line = Perfect detection on all reachable targets in FOV',
                    fontsize=14, fontweight='bold', color='darkgreen')
    else:
        ax.set_title('F1-Score: Visible Targets in FOV (Glass Ceiling Test)',
                    fontsize=14, fontweight='bold', color='darkgreen')
    
    ax.set_xlabel('Radar Pulse Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score (Visible Pixels Only)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_visible_trace_glass_ceiling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ F1 Visible (Glass Ceiling) trace saved")


def plot_iou_visible_traces(all_results, save_dir):
    """
    Plot IoU on VISIBLE pixels only - Glass Ceiling Test
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c', 'camera_only': '#95a5a6'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random', 'camera_only': 'Camera Only'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot camera-only baseline if available
    if 'camera_only' in all_results:
        aggregated_cam, _ = all_results['camera_only']
        iou_cam = aggregated_cam['iou_visible']
        ax.axhline(y=iou_cam, color=colors['camera_only'], linestyle=':', 
                  linewidth=3, alpha=0.6, label=labels['camera_only'], zorder=1)
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        iou_trace = results_dict['iou_visible_trace']
        
        max_len = max(len(trace) for trace in iou_trace)
        padded_traces = []
        for trace in iou_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        if strategy == 'entropy':
            ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=4, 
                   label=labels[strategy], marker='o', markersize=8, alpha=0.95, zorder=3)
        else:
            ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
                   label=labels[strategy], marker='s', markersize=6, alpha=0.85, zorder=2,
                   linestyle='--')
        
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.15)
    
    # THEORETICAL MAXIMUM
    ax.axhline(y=1.0, color='gold', linestyle='-.', linewidth=4, alpha=0.8, 
              label='🏆 Theoretical Maximum', zorder=4)
    
    ax.set_xlabel('Radar Pulse Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('IoU (Visible in FOV)', fontsize=13, fontweight='bold')
    ax.set_title('IoU: Visible Targets in FOV - Glass Ceiling Test\n' +
                'Gold line = Perfect overlap on reachable targets in radar FOV',
                fontsize=14, fontweight='bold', color='darkgreen')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_visible_trace_glass_ceiling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ IoU Visible (Glass Ceiling) trace saved")


def plot_error_visible_traces(all_results, save_dir):
    """
    Plot MAE on VISIBLE pixels only - Glass Ceiling Test (LOWER IS BETTER)
    """
    strategies = [k for k in list(all_results.keys()) if k != 'camera_only']
    colors = {'entropy': '#2ecc71', 'uniform': '#3498db', 'random': '#e74c3c', 'camera_only': '#95a5a6'}
    labels = {'entropy': 'Cognitive', 'uniform': 'Uniform', 'random': 'Random', 'camera_only': 'Camera Only'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot camera-only baseline if available
    if 'camera_only' in all_results:
        aggregated_cam, _ = all_results['camera_only']
        error_cam = aggregated_cam['error_visible']
        ax.axhline(y=error_cam, color=colors['camera_only'], linestyle=':', 
                  linewidth=3, alpha=0.6, label=labels['camera_only'], zorder=1)
    
    for strategy in strategies:
        aggregated, results_dict = all_results[strategy]
        error_trace = results_dict['error_visible_trace']
        
        max_len = max(len(trace) for trace in error_trace)
        padded_traces = []
        for trace in error_trace:
            if len(trace) < max_len:
                padded = list(trace) + [trace[-1]] * (max_len - len(trace))
            else:
                padded = trace
            padded_traces.append(padded)
        
        avg_trace = np.mean(padded_traces, axis=0)
        std_trace = np.std(padded_traces, axis=0)
        pulses = np.arange(len(avg_trace))
        
        if strategy == 'entropy':
            ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=4, 
                   label=labels[strategy], marker='o', markersize=8, alpha=0.95, zorder=3)
        else:
            ax.plot(pulses, avg_trace, color=colors[strategy], linewidth=3, 
                   label=labels[strategy], marker='s', markersize=6, alpha=0.85, zorder=2,
                   linestyle='--')
        
        ax.fill_between(pulses, avg_trace - std_trace, avg_trace + std_trace,
                        color=colors[strategy], alpha=0.15)
    
    # THEORETICAL MINIMUM (zero error on visible pixels)
    ax.axhline(y=0.0, color='gold', linestyle='-.', linewidth=4, alpha=0.8, 
              label='🏆 Theoretical Minimum (0.0)', zorder=4)
    ax.fill_between([0, max_len-1], [0.0, 0.0], [0.05, 0.05], 
                    color='gold', alpha=0.1, label='Near-Perfect Zone')
    
    ax.set_xlabel('Radar Pulse Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAE (Visible in FOV) ⬇ LOWER IS BETTER', fontsize=13, fontweight='bold')
    ax.set_title('Error: Visible Targets in FOV - Glass Ceiling Test\n' +
                'Gold line = Zero error on reachable targets in radar FOV',
                fontsize=14, fontweight='bold', color='darkgreen')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_visible_trace_glass_ceiling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Error Visible (Glass Ceiling) trace saved")


def visualize_visibility_mask_diagnostic(ground_truth, visibility_mask, camera_prior, 
                                         radar_fused, scene_idx, save_dir, fov_mask=None):
    """
    DIAGNOSTIC: Visualize what the visibility mask looks like
    
    Shows:
    1. Ground truth (what actually exists)
    2. Visibility mask (what's physically reachable from ego WITHIN FOV)
    3. Occluded regions (behind walls/cars - NOT visible)
    4. Performance comparison: Global vs. Visible-only
    
    NOTE: visibility_mask should already be the intersection of:
          - Physically reachable (not occluded) 
          - Within radar FOV (120° arc)
    
    This helps debug if visibility calculation is correct.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    H, W = ground_truth.shape
    
    # === ROW 1: GEOMETRY ===
    
    # 1. Ground Truth
    ax1 = axes[0, 0]
    ax1.imshow(ground_truth, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax1.set_title('Ground Truth\n(All objects in scene)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (grid)')
    ax1.set_ylabel('Y (grid)')
    occupied_pixels = (ground_truth > 0.5).sum()
    ax1.text(0.02, 0.98, f'Occupied: {occupied_pixels} px', 
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Visibility Mask (Visible + In FOV)
    ax2 = axes[0, 1]
    ax2.imshow(visibility_mask, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    ax2.set_title('GLASS CEILING Mask\n(Visible AND in FOV)', 
                 fontsize=14, fontweight='bold', color='green')
    ax2.set_xlabel('X (grid)')
    ax2.set_ylabel('Y (grid)')
    # Add ego position marker
    ax2.plot(W//2, H//2, 'b*', markersize=20, label='Ego Vehicle')
    ax2.legend(loc='upper right')
    visible_pixels = visibility_mask.sum()
    visibility_ratio = visible_pixels / (H * W)
    ax2.text(0.02, 0.98, f'Visible + FOV: {int(visible_pixels)} px\n{visibility_ratio*100:.1f}% of map', 
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. Occluded Regions (Ground Truth - Visible)
    ax3 = axes[0, 2]
    gt_binary = (ground_truth > 0.5)
    visible_bool = visibility_mask.astype(bool)
    occluded_targets = gt_binary & (~visible_bool)
    visible_targets = gt_binary & visible_bool
    
    # Create RGB overlay
    overlay = np.zeros((H, W, 3))
    overlay[visible_targets, 1] = 1.0  # Green = visible targets
    overlay[occluded_targets, 0] = 1.0  # Red = occluded targets (behind obstacles)
    overlay[~gt_binary & visible_bool, 2] = 0.3  # Blue = visible free space
    
    ax3.imshow(overlay, origin='lower')
    ax3.set_title('Target Reachability Analysis\n🟢 Visible | 🔴 Occluded', 
                 fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (grid)')
    ax3.set_ylabel('Y (grid)')
    
    visible_target_count = visible_targets.sum()
    occluded_target_count = occluded_targets.sum()
    if occupied_pixels > 0:
        visible_target_ratio = visible_target_count / occupied_pixels
        occluded_target_ratio = occluded_target_count / occupied_pixels
    else:
        visible_target_ratio = 0.0
        occluded_target_ratio = 0.0
    
    ax3.text(0.02, 0.98, 
            f'🟢 Visible targets: {visible_target_count} ({visible_target_ratio*100:.1f}%)\n' +
            f'🔴 Occluded targets: {occluded_target_count} ({occluded_target_ratio*100:.1f}%)',
            transform=ax3.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === ROW 2: PERFORMANCE ===
    
    # 4. Camera Prior (all pixels)
    ax4 = axes[1, 0]
    ax4.imshow(camera_prior, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax4.contour(ground_truth, levels=[0.5], colors='red', linewidths=2, alpha=0.7)
    ax4.set_title('Camera Prior\n(Red = GT boundary)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X (grid)')
    ax4.set_ylabel('Y (grid)')
    
    from radar_simulation import compute_segmentation_metrics
    cam_metrics_global = compute_segmentation_metrics(camera_prior, ground_truth)
    ax4.text(0.02, 0.98, f'Global F1: {cam_metrics_global["f1_score"]:.3f}',
            transform=ax4.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 5. Radar Fused (all pixels)
    ax5 = axes[1, 1]
    ax5.imshow(radar_fused, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax5.contour(ground_truth, levels=[0.5], colors='red', linewidths=2, alpha=0.7)
    ax5.set_title('Radar Fused\n(Red = GT boundary)', fontsize=14, fontweight='bold', color='green')
    ax5.set_xlabel('X (grid)')
    ax5.set_ylabel('Y (grid)')
    
    fused_metrics_global = compute_segmentation_metrics(radar_fused, ground_truth)
    ax5.text(0.02, 0.98, f'Global F1: {fused_metrics_global["f1_score"]:.3f}',
            transform=ax5.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 6. Performance Comparison: Global vs. Visible-Only
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Compute visible-only metrics
    from radar_simulation import compute_visible_metrics
    cam_visible = compute_visible_metrics(camera_prior, ground_truth, visibility_mask)
    fused_visible = compute_visible_metrics(radar_fused, ground_truth, visibility_mask)
    
    comparison_text = f"""
PERFORMANCE COMPARISON
{'='*40}

📊 CAMERA PRIOR:
  Global F1:   {cam_metrics_global['f1_score']:.4f}
  Visible F1:  {cam_visible['f1_visible']:.4f}
  Difference:  {cam_visible['f1_visible'] - cam_metrics_global['f1_score']:+.4f}

📡 RADAR FUSED:
  Global F1:   {fused_metrics_global['f1_score']:.4f}
  Visible F1:  {fused_visible['f1_visible']:.4f}
  Difference:  {fused_visible['f1_visible'] - fused_metrics_global['f1_score']:+.4f}

{'='*40}
🎯 TARGET REACHABILITY:
  Total targets:     {occupied_pixels}
  Visible targets:   {visible_target_count} ({visible_target_ratio*100:.1f}%)
  Occluded targets:  {occluded_target_count} ({occluded_target_ratio*100:.1f}%)

📍 GLASS CEILING (Visible + In FOV):
  Total pixels:      {H * W}
  Visible+FOV px:    {int(visible_pixels)} ({visibility_ratio*100:.1f}%)
  Outside glass:     {H * W - int(visible_pixels)} ({(1-visibility_ratio)*100:.1f}%)

{'='*40}
⚠️ EXPECTED BEHAVIOR:
  Glass Ceiling = Visible AND in FOV
  (Radar can only scan forward 120° arc)
  
  If glass ceiling = 30-40% of map, then:
    Visible F1 should be MUCH higher than Global F1
    (evaluating only on radar's scannable region)
  
  Visible F1 approaching 1.0 = Near-perfect
  detection on all reachable targets in FOV
"""
    
    ax6.text(0.05, 0.95, comparison_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle(f'GLASS CEILING DIAGNOSTIC - Scene {scene_idx}\n' +
                f'Evaluating: Visible Targets Within FOV (Radar\'s Achievable Region)',
                fontsize=16, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'visibility_diagnostic_scene_{scene_idx:02d}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🔍 Visibility diagnostic saved for scene {scene_idx}")
