"""
Comprehensive Entropy Validation Test Suite
Compares baseline LSS model vs trained ensemble across 5 key uncertainty scenarios
"""

import torch
import torch.nn as nn
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF
import PIL.Image
from tqdm import tqdm

# === CONFIGURATION ===
# Note: Paths are relative to parent directory (repos/)
BASELINE_PATH = '../checkpoints/model525000.pt'
ENSEMBLE_PATH = '../checkpoints/runs/mini/hybrid_entropy_new/2025-12-27_22-09-42/checkpoints/model_best.pth'
DATAROOT = '../data/'
NUM_TEST_SCENES = 100  # Number of scenes to analyze
NUM_EXAMPLES_TO_SAVE = 5  # Number of example images to save per test
# =====================

def get_model_name_from_path(model_path):
    """Extract a readable model name from checkpoint path"""
    # e.g., "checkpoints/runs/mini/hybrid_entropy_new/2025-12-27_22-09-42/checkpoints/model_best.pth"
    # -> "hybrid_entropy_new_2025-12-27_22-09-42"
    parts = model_path.split('/')
    if 'runs' in parts:
        runs_idx = parts.index('runs')
        if len(parts) > runs_idx + 3:
            return f"{parts[runs_idx + 2]}_{parts[runs_idx + 3]}"
    return "unknown_model"

def setup_results_dir(model_path):
    """Create organized results directory for this model"""
    model_name = get_model_name_from_path(model_path)
    results_dir = f'entropy_validation_results_{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'examples'), exist_ok=True)
    return results_dir

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Add parent directory's lift-splat-shoot to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

# ==========================================
#              MODEL CLASSES
# ==========================================

class BaselineLSS(nn.Module):
    """Single pretrained LSS model"""
    def __init__(self, grid_conf, data_aug_conf, outC=1):
        super().__init__()
        self.model = LiftSplatShoot(grid_conf, data_aug_conf, outC=outC)
    
    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        return self.model(x, rots, trans, intrins, post_rots, post_trans).unsqueeze(1)

class EnsembleLSS(nn.Module):
    """5-head ensemble model"""
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
#           UTILITY FUNCTIONS
# ==========================================

def compute_entropy(probs):
    """Compute binary entropy from probabilities"""
    eps = 1e-7
    entropy = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
    return entropy

def compute_brightness(img_batch):
    """Compute average brightness of image batch (B, N, C, H, W)"""
    # Average across all dimensions except batch
    brightness = img_batch.mean(dim=(1, 2, 3, 4))
    return brightness

def apply_occlusion(imgs, cam_idx=1, occlusion_type='center'):
    """Apply occlusion to specified camera"""
    imgs_occluded = imgs.clone()
    C, H, W = imgs.shape[2], imgs.shape[3], imgs.shape[4]
    
    if occlusion_type == 'center':
        # Occlude center region
        h_start, h_end = int(H*0.3), int(H*0.7)
        w_start, w_end = int(W*0.3), int(W*0.7)
        imgs_occluded[:, cam_idx, :, h_start:h_end, w_start:w_end] = 0
    
    return imgs_occluded

def compute_complexity(gt_binimg):
    """
    Compute scene complexity score based on:
    - Number of occupied cells
    - Edge density (structural complexity)
    """
    occupied = gt_binimg.sum().item()
    
    # Sobel edge detection
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    gt_padded = torch.nn.functional.pad(gt_binimg.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1), mode='replicate')
    edges_x = torch.nn.functional.conv2d(gt_padded, kernel_x)
    edges_y = torch.nn.functional.conv2d(gt_padded, kernel_y)
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    edge_density = edges.sum().item()
    
    return occupied + edge_density * 0.1  # Weighted combination

def get_distance_mask(bev_shape, distance_range, grid_conf):
    """Create binary mask for specific distance range in BEV coordinates"""
    H, W = bev_shape
    center_y = H // 2
    center_x = W // 2
    
    # Create distance map from ego vehicle
    y_coords = torch.arange(H).view(-1, 1).expand(H, W)
    x_coords = torch.arange(W).view(1, -1).expand(H, W)
    
    # Convert to meters (based on grid_conf)
    x_bound = grid_conf['xbound']
    y_bound = grid_conf['ybound']
    resolution = x_bound[2]  # 0.5m per pixel
    
    # Distance from center in meters
    dy = (y_coords - center_y) * resolution
    dx = (x_coords - center_x) * resolution
    distances = torch.sqrt(dx**2 + dy**2)
    
    # Create mask for distance range
    mask = (distances >= distance_range[0]) & (distances < distance_range[1])
    return mask.float()

def denormalize_img(img_tensor):
    """Denormalize image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

# ==========================================
#              TEST FUNCTIONS
# ==========================================

def test_1_darkness_sensitivity(baseline_model, ensemble_model, val_loader, device, save_dir):
    """
    Test 1: Darkness Sensitivity
    Hypothesis: Darker scenes → higher entropy, ensemble shows stronger correlation
    """
    print("\n" + "="*60)
    print("TEST 1: DARKNESS SENSITIVITY")
    print("="*60)
    
    baseline_model.eval()
    ensemble_model.eval()
    
    brightness_list = []
    baseline_entropy_list = []
    ensemble_entropy_list = []
    all_data = []  # Store data for example saving
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Analyzing scenes")):
            if i >= NUM_TEST_SCENES:
                break
            
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            imgs = imgs.to(device)
            rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device)
            post_rots, post_trans = post_rots.to(device), post_trans.to(device)
            
            # Compute brightness
            brightness = compute_brightness(imgs).item()
            brightness_list.append(brightness)
            
            # Baseline prediction
            base_pred = baseline_model(imgs, rots, trans, intrins, post_rots, post_trans)
            base_prob = torch.sigmoid(base_pred[:, 0])
            base_entropy = compute_entropy(base_prob).mean().item()
            baseline_entropy_list.append(base_entropy)
            
            # Ensemble prediction
            ens_preds = ensemble_model(imgs, rots, trans, intrins, post_rots, post_trans)
            ens_probs = torch.sigmoid(ens_preds)
            ens_mean_prob = ens_probs.mean(dim=1)[:, 0]
            ens_entropy = compute_entropy(ens_mean_prob).mean().item()
            ensemble_entropy_list.append(ens_entropy)
            
            # Store for example saving
            all_data.append({
                'imgs': imgs.cpu(),
                'base_prob': base_prob.cpu(),
                'base_entropy_map': compute_entropy(base_prob).cpu(),
                'ens_prob': ens_mean_prob.cpu(),
                'ens_entropy_map': compute_entropy(ens_mean_prob).cpu(),
                'gt': binimgs.cpu(),
                'brightness': brightness
            })
    
    # Statistical analysis
    baseline_corr, baseline_p = pearsonr(brightness_list, baseline_entropy_list)
    ensemble_corr, ensemble_p = pearsonr(brightness_list, ensemble_entropy_list)
    
    # Sort by brightness for better visualization
    sorted_idx = np.argsort(brightness_list)
    brightness_sorted = np.array(brightness_list)[sorted_idx]
    baseline_sorted = np.array(baseline_entropy_list)[sorted_idx]
    ensemble_sorted = np.array(ensemble_entropy_list)[sorted_idx]
    
    # Find darkest and brightest scenes
    darkest_10 = sorted_idx[:len(sorted_idx)//10]
    brightest_10 = sorted_idx[-len(sorted_idx)//10:]
    
    dark_base_mean = np.mean([baseline_entropy_list[i] for i in darkest_10])
    dark_ens_mean = np.mean([ensemble_entropy_list[i] for i in darkest_10])
    bright_base_mean = np.mean([baseline_entropy_list[i] for i in brightest_10])
    bright_ens_mean = np.mean([ensemble_entropy_list[i] for i in brightest_10])
    
    # T-test comparing dark vs bright
    base_ttest = stats.ttest_ind(
        [baseline_entropy_list[i] for i in darkest_10],
        [baseline_entropy_list[i] for i in brightest_10]
    )
    ens_ttest = stats.ttest_ind(
        [ensemble_entropy_list[i] for i in darkest_10],
        [ensemble_entropy_list[i] for i in brightest_10]
    )
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    axes[0].scatter(brightness_sorted, baseline_sorted, alpha=0.5, label=f'Baseline (r={baseline_corr:.3f})', s=30)
    axes[0].scatter(brightness_sorted, ensemble_sorted, alpha=0.5, label=f'Ensemble (r={ensemble_corr:.3f})', s=30)
    axes[0].set_xlabel('Scene Brightness', fontsize=12)
    axes[0].set_ylabel('Mean Entropy', fontsize=12)
    axes[0].set_title('Brightness vs Entropy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Comparison bar plot
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [dark_base_mean, bright_base_mean], width, label='Baseline', alpha=0.8)
    axes[1].bar(x + width/2, [dark_ens_mean, bright_ens_mean], width, label='Ensemble', alpha=0.8)
    axes[1].set_ylabel('Mean Entropy', fontsize=12)
    axes[1].set_title('Dark vs Bright Scenes', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Darkest 10%', 'Brightest 10%'])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'darkness_comparison.png'), dpi=150)
    plt.close()
    
    # Save statistics
    stats_dict = {
        'baseline_correlation': float(baseline_corr),
        'baseline_p_value': float(baseline_p),
        'ensemble_correlation': float(ensemble_corr),
        'ensemble_p_value': float(ensemble_p),
        'dark_baseline_mean': float(dark_base_mean),
        'dark_ensemble_mean': float(dark_ens_mean),
        'bright_baseline_mean': float(bright_base_mean),
        'bright_ensemble_mean': float(bright_ens_mean),
        'baseline_ttest_statistic': float(base_ttest.statistic),
        'baseline_ttest_pvalue': float(base_ttest.pvalue),
        'ensemble_ttest_statistic': float(ens_ttest.statistic),
        'ensemble_ttest_pvalue': float(ens_ttest.pvalue),
    }
    
    with open(os.path.join(save_dir, 'darkness_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    # ===== SAVE COMPREHENSIVE EXAMPLE IMAGES =====
    print(f"\n💾 Saving comprehensive example images...")
    examples_dir = os.path.join(save_dir, 'examples')
    
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    # Select darkest and brightest examples
    darkest_indices = sorted_idx[:NUM_EXAMPLES_TO_SAVE]
    brightest_indices = sorted_idx[-NUM_EXAMPLES_TO_SAVE:]
    
    def save_comprehensive_example(data, filename, scene_type, scene_num):
        """Create comprehensive visualization with all 6 cameras + predictions + entropy"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
        
        # === ROW 1: ALL 6 CAMERAS ===
        for cam_idx in range(6):
            ax = fig.add_subplot(gs[0, cam_idx])
            ax.imshow(denormalize_img(data['imgs'][0, cam_idx]))
            ax.set_title(cams[cam_idx], fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Add orientation guide
        fig.text(0.5, 0.67, '← LEFT SIDE | VEHICLE | RIGHT SIDE →', 
                ha='center', fontsize=11, style='italic', color='gray')
        
        # === ROW 2: GROUND TRUTH & PREDICTIONS ===
        # Note: Flip Y-axis (left-right) to match camera orientation
        # In NuScenes BEV: Y-axis points LEFT, but when displayed we want it to match cameras
        
        # Ground Truth
        ax_gt = fig.add_subplot(gs[1, 0:2])
        gt_2d = data['gt'][0, 0].numpy()
        gt_2d_flipped = np.fliplr(gt_2d)  # Flip left-right to match cameras
        ax_gt.imshow(gt_2d_flipped, cmap='binary_r', origin='lower')
        ax_gt.set_title("Ground Truth\n(BEV: Bird's Eye View)", fontsize=12, fontweight='bold')
        ax_gt.set_xlabel("← LEFT | RIGHT →", fontsize=9, style='italic')
        ax_gt.set_ylabel("FORWARD ↑", fontsize=9, style='italic')
        
        # Baseline Prediction - squeeze to 2D
        ax_base_pred = fig.add_subplot(gs[1, 2:4])
        base_prob_np = data['base_prob'].numpy()
        base_prob_2d = base_prob_np.squeeze()
        base_prob_2d_flipped = np.fliplr(base_prob_2d)  # Flip to match cameras
        im_bp = ax_base_pred.imshow(base_prob_2d_flipped, cmap='jet', origin='lower', vmin=0, vmax=1)
        ax_base_pred.set_title("Baseline Prediction", fontsize=12, fontweight='bold')
        ax_base_pred.set_xlabel("← LEFT | RIGHT →", fontsize=9, style='italic')
        plt.colorbar(im_bp, ax=ax_base_pred, fraction=0.046, pad=0.04)
        
        # Ensemble Prediction - squeeze to 2D
        ax_ens_pred = fig.add_subplot(gs[1, 4:6])
        ens_prob_np = data['ens_prob'].numpy()
        ens_prob_2d = ens_prob_np.squeeze()
        ens_prob_2d_flipped = np.fliplr(ens_prob_2d)  # Flip to match cameras
        im_ep = ax_ens_pred.imshow(ens_prob_2d_flipped, cmap='jet', origin='lower', vmin=0, vmax=1)
        ax_ens_pred.set_title("Ensemble Prediction", fontsize=12, fontweight='bold')
        ax_ens_pred.set_xlabel("← LEFT | RIGHT →", fontsize=9, style='italic')
        plt.colorbar(im_ep, ax=ax_ens_pred, fraction=0.046, pad=0.04)
        
        # === ROW 3: ENTROPY MAPS ===
        # Baseline Entropy - squeeze to 2D
        ax_base_ent = fig.add_subplot(gs[2, 1:3])
        base_ent_np = data['base_entropy_map'].numpy()
        base_ent_2d = base_ent_np.squeeze()
        base_ent_2d_flipped = np.fliplr(base_ent_2d)  # Flip to match cameras
        im_be = ax_base_ent.imshow(base_ent_2d_flipped, cmap='inferno', origin='lower')
        ax_base_ent.set_title("Baseline Entropy\n(Uncertainty Map)", fontsize=12, fontweight='bold')
        ax_base_ent.set_xlabel("← LEFT | RIGHT →", fontsize=9, style='italic')
        plt.colorbar(im_be, ax=ax_base_ent, fraction=0.046, pad=0.04)
        
        # Ensemble Entropy - squeeze to 2D
        ax_ens_ent = fig.add_subplot(gs[2, 3:5])
        ens_ent_np = data['ens_entropy_map'].numpy()
        ens_ent_2d = ens_ent_np.squeeze()
        ens_ent_2d_flipped = np.fliplr(ens_ent_2d)  # Flip to match cameras
        im_ee = ax_ens_ent.imshow(ens_ent_2d_flipped, cmap='inferno', origin='lower')
        ax_ens_ent.set_title("Ensemble Entropy\n(Uncertainty Map)", fontsize=12, fontweight='bold')
        ax_ens_ent.set_xlabel("← LEFT | RIGHT →", fontsize=9, style='italic')
        plt.colorbar(im_ee, ax=ax_ens_ent, fraction=0.046, pad=0.04)
        
        # Main title with better context
        if scene_type == "Dark":
            title = f"Low-Light Scene #{scene_num} (Darkest in Dataset) | Normalized Brightness: {data['brightness']:.3f}"
        else:
            title = f"Well-Lit Scene #{scene_num} (Brightest in Dataset) | Normalized Brightness: {data['brightness']:.3f}"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save darkest (low-light) scenes
    for idx, scene_idx in enumerate(darkest_indices):
        data = all_data[scene_idx]
        filename = os.path.join(examples_dir, f'lowlight_scene_{idx+1}.png')
        save_comprehensive_example(data, filename, "Dark", idx+1)
    
    # Save brightest (well-lit) scenes
    for idx, scene_idx in enumerate(brightest_indices):
        data = all_data[scene_idx]
        filename = os.path.join(examples_dir, f'welllit_scene_{idx+1}.png')
        save_comprehensive_example(data, filename, "Bright", idx+1)
    
    print(f"   Saved {NUM_EXAMPLES_TO_SAVE} low-light and {NUM_EXAMPLES_TO_SAVE} well-lit comprehensive examples")
    print(f"   Note: 'Low-light' = relatively darker scenes in dataset (may still be daytime)")
    print(f"         Brightness range: {brightness_sorted[0]:.3f} to {brightness_sorted[-1]:.3f}")
    
    print(f"\n✅ Test 1 Complete!")
    print(f"   Baseline: r={baseline_corr:.3f}, p={baseline_p:.4f}")
    print(f"   Ensemble: r={ensemble_corr:.3f}, p={ensemble_p:.4f}")
    print(f"   {'✓ Ensemble shows stronger correlation' if abs(ensemble_corr) > abs(baseline_corr) else '✗ Baseline stronger'}")
    
    return stats_dict

def test_2_distance_uncertainty(baseline_model, ensemble_model, val_loader, device, grid_conf, save_dir):
    """
    Test 2: Distance-Based Uncertainty
    Hypothesis: Far objects have higher entropy than near objects
    """
    print("\n" + "="*60)
    print("TEST 2: DISTANCE-BASED UNCERTAINTY")
    print("="*60)
    
    baseline_model.eval()
    ensemble_model.eval()
    
    distance_bands = [(0, 15), (15, 30), (30, 45)]
    band_names = ['Near (0-15m)', 'Mid (15-30m)', 'Far (30-45m)']
    
    baseline_by_distance = {name: [] for name in band_names}
    ensemble_by_distance = {name: [] for name in band_names}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Analyzing distance bands")):
            if i >= NUM_TEST_SCENES:
                break
            
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            imgs = imgs.to(device)
            rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device)
            post_rots, post_trans = post_rots.to(device), post_trans.to(device)
            
            # Baseline
            base_pred = baseline_model(imgs, rots, trans, intrins, post_rots, post_trans)
            base_prob = torch.sigmoid(base_pred[0, 0, 0]).cpu()
            base_entropy = compute_entropy(base_prob)
            
            # Ensemble
            ens_preds = ensemble_model(imgs, rots, trans, intrins, post_rots, post_trans)
            ens_probs = torch.sigmoid(ens_preds)
            ens_mean_prob = ens_probs.mean(dim=1)[0, 0].cpu()
            ens_entropy = compute_entropy(ens_mean_prob)
            
            # Compute entropy per distance band
            for band, name in zip(distance_bands, band_names):
                mask = get_distance_mask(base_entropy.shape, band, grid_conf)
                if mask.sum() > 0:
                    baseline_by_distance[name].append(base_entropy[mask > 0].mean().item())
                    ensemble_by_distance[name].append(ens_entropy[mask > 0].mean().item())
    
    # Statistical analysis
    baseline_means = [np.mean(baseline_by_distance[name]) for name in band_names]
    ensemble_means = [np.mean(ensemble_by_distance[name]) for name in band_names]
    baseline_stds = [np.std(baseline_by_distance[name]) for name in band_names]
    ensemble_stds = [np.std(ensemble_by_distance[name]) for name in band_names]
    
    # ANOVA test
    baseline_anova = stats.f_oneway(*[baseline_by_distance[name] for name in band_names])
    ensemble_anova = stats.f_oneway(*[ensemble_by_distance[name] for name in band_names])
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    data_baseline = [baseline_by_distance[name] for name in band_names]
    data_ensemble = [ensemble_by_distance[name] for name in band_names]
    
    positions = np.arange(len(band_names))
    bp1 = axes[0].boxplot(data_baseline, positions=positions - 0.2, widths=0.35, patch_artist=True,
                           boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue', linewidth=2))
    bp2 = axes[0].boxplot(data_ensemble, positions=positions + 0.2, widths=0.35, patch_artist=True,
                           boxprops=dict(facecolor='lightcoral'), medianprops=dict(color='red', linewidth=2))
    
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(band_names)
    axes[0].set_ylabel('Entropy', fontsize=12)
    axes[0].set_title('Entropy by Distance Band', fontsize=14, fontweight='bold')
    axes[0].legend([bp1["boxes"][0], bp2["boxes"][0]], ['Baseline', 'Ensemble'], loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Mean comparison
    x = np.arange(len(band_names))
    width = 0.35
    axes[1].bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline', alpha=0.8, capsize=5)
    axes[1].bar(x + width/2, ensemble_means, width, yerr=ensemble_stds, label='Ensemble', alpha=0.8, capsize=5)
    axes[1].set_ylabel('Mean Entropy', fontsize=12)
    axes[1].set_title('Mean Entropy by Distance', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(band_names)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distance_boxplot.png'), dpi=150)
    plt.close()
    
    # Save statistics
    stats_dict = {
        'baseline_means': [float(m) for m in baseline_means],
        'ensemble_means': [float(m) for m in ensemble_means],
        'baseline_stds': [float(s) for s in baseline_stds],
        'ensemble_stds': [float(s) for s in ensemble_stds],
        'baseline_anova_f': float(baseline_anova.statistic),
        'baseline_anova_p': float(baseline_anova.pvalue),
        'ensemble_anova_f': float(ensemble_anova.statistic),
        'ensemble_anova_p': float(ensemble_anova.pvalue),
    }
    
    with open(os.path.join(save_dir, 'distance_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"\n✅ Test 2 Complete!")
    print(f"   Baseline ANOVA: F={baseline_anova.statistic:.2f}, p={baseline_anova.pvalue:.4f}")
    print(f"   Ensemble ANOVA: F={ensemble_anova.statistic:.2f}, p={ensemble_anova.pvalue:.4f}")
    print(f"   Increasing entropy with distance: {'✓ YES' if ensemble_means[-1] > ensemble_means[0] else '✗ NO'}")
    
    return stats_dict

def test_3_occlusion_response(baseline_model, ensemble_model, val_loader, device, save_dir):
    """
    Test 3: Occlusion Response
    Hypothesis: Occluding cameras increases entropy in affected BEV regions
    """
    print("\n" + "="*60)
    print("TEST 3: OCCLUSION RESPONSE")
    print("="*60)
    
    baseline_model.eval()
    ensemble_model.eval()
    
    baseline_delta_list = []
    ensemble_delta_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Testing occlusion")):
            if i >= NUM_TEST_SCENES:
                break
            
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            imgs = imgs.to(device)
            rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device)
            post_rots, post_trans = post_rots.to(device), post_trans.to(device)
            
            # Clean predictions
            base_clean = baseline_model(imgs, rots, trans, intrins, post_rots, post_trans)
            ens_clean = ensemble_model(imgs, rots, trans, intrins, post_rots, post_trans)
            
            base_prob_clean = torch.sigmoid(base_clean[0, 0, 0])
            ens_prob_clean = torch.sigmoid(ens_clean).mean(dim=1)[0, 0]
            
            base_ent_clean = compute_entropy(base_prob_clean)
            ens_ent_clean = compute_entropy(ens_prob_clean)
            
            # Occluded predictions (front camera)
            imgs_occ = apply_occlusion(imgs, cam_idx=1)
            base_occ = baseline_model(imgs_occ, rots, trans, intrins, post_rots, post_trans)
            ens_occ = ensemble_model(imgs_occ, rots, trans, intrins, post_rots, post_trans)
            
            base_prob_occ = torch.sigmoid(base_occ[0, 0, 0])
            ens_prob_occ = torch.sigmoid(ens_occ).mean(dim=1)[0, 0]
            
            base_ent_occ = compute_entropy(base_prob_occ)
            ens_ent_occ = compute_entropy(ens_prob_occ)
            
            # Compute delta in front region (where occlusion should matter most)
            H, W = base_ent_clean.shape
            front_region = slice(H//2, H), slice(W//3, 2*W//3)
            
            base_delta = (base_ent_occ[front_region] - base_ent_clean[front_region]).mean().item()
            ens_delta = (ens_ent_occ[front_region] - ens_ent_clean[front_region]).mean().item()
            
            baseline_delta_list.append(base_delta)
            ensemble_delta_list.append(ens_delta)
    
    # Statistical analysis
    baseline_mean_delta = np.mean(baseline_delta_list)
    ensemble_mean_delta = np.mean(ensemble_delta_list)
    baseline_std_delta = np.std(baseline_delta_list)
    ensemble_std_delta = np.std(ensemble_delta_list)
    
    # T-test: is delta significantly different from zero?
    base_ttest = stats.ttest_1samp(baseline_delta_list, 0)
    ens_ttest = stats.ttest_1samp(ensemble_delta_list, 0)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of deltas
    axes[0].hist(baseline_delta_list, bins=30, alpha=0.6, label=f'Baseline (μ={baseline_mean_delta:.4f})', edgecolor='black')
    axes[0].hist(ensemble_delta_list, bins=30, alpha=0.6, label=f'Ensemble (μ={ensemble_mean_delta:.4f})', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[0].set_xlabel('Entropy Change (Δ)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Entropy Change Due to Occlusion', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot comparison
    axes[1].boxplot([baseline_delta_list, ensemble_delta_list], labels=['Baseline', 'Ensemble'],
                     patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Entropy Change (Δ)', fontsize=12)
    axes[1].set_title('Occlusion Sensitivity Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'occlusion_heatmap.png'), dpi=150)
    plt.close()
    
    # Save statistics
    stats_dict = {
        'baseline_mean_delta': float(baseline_mean_delta),
        'baseline_std_delta': float(baseline_std_delta),
        'ensemble_mean_delta': float(ensemble_mean_delta),
        'ensemble_std_delta': float(ensemble_std_delta),
        'baseline_ttest_statistic': float(base_ttest.statistic),
        'baseline_ttest_pvalue': float(base_ttest.pvalue),
        'ensemble_ttest_statistic': float(ens_ttest.statistic),
        'ensemble_ttest_pvalue': float(ens_ttest.pvalue),
    }
    
    with open(os.path.join(save_dir, 'occlusion_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"\n✅ Test 3 Complete!")
    print(f"   Baseline: Δ={baseline_mean_delta:.4f}, p={base_ttest.pvalue:.4f}")
    print(f"   Ensemble: Δ={ensemble_mean_delta:.4f}, p={ens_ttest.pvalue:.4f}")
    print(f"   {'✓ Positive entropy increase' if ensemble_mean_delta > 0 else '✗ No increase'}")
    
    return stats_dict

def test_4_complexity_correlation(baseline_model, ensemble_model, val_loader, device, save_dir):
    """
    Test 4: Scene Complexity Correlation
    Hypothesis: Complex scenes have higher entropy
    """
    print("\n" + "="*60)
    print("TEST 4: SCENE COMPLEXITY CORRELATION")
    print("="*60)
    
    baseline_model.eval()
    ensemble_model.eval()
    
    complexity_list = []
    baseline_entropy_list = []
    ensemble_entropy_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Analyzing complexity")):
            if i >= NUM_TEST_SCENES:
                break
            
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            imgs = imgs.to(device)
            rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device)
            post_rots, post_trans = post_rots.to(device), post_trans.to(device)
            
            # Compute complexity
            complexity = compute_complexity(binimgs[0, 0])
            complexity_list.append(complexity)
            
            # Baseline
            base_pred = baseline_model(imgs, rots, trans, intrins, post_rots, post_trans)
            base_prob = torch.sigmoid(base_pred[:, 0])
            base_entropy = compute_entropy(base_prob).mean().item()
            baseline_entropy_list.append(base_entropy)
            
            # Ensemble
            ens_preds = ensemble_model(imgs, rots, trans, intrins, post_rots, post_trans)
            ens_probs = torch.sigmoid(ens_preds)
            ens_mean_prob = ens_probs.mean(dim=1)[:, 0]
            ens_entropy = compute_entropy(ens_mean_prob).mean().item()
            ensemble_entropy_list.append(ens_entropy)
    
    # Statistical analysis
    baseline_corr, baseline_p = pearsonr(complexity_list, baseline_entropy_list)
    ensemble_corr, ensemble_p = pearsonr(complexity_list, ensemble_entropy_list)
    
    # Linear regression
    baseline_slope, baseline_intercept = np.polyfit(complexity_list, baseline_entropy_list, 1)
    ensemble_slope, ensemble_intercept = np.polyfit(complexity_list, ensemble_entropy_list, 1)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter with regression
    axes[0].scatter(complexity_list, baseline_entropy_list, alpha=0.5, label='Baseline', s=30)
    axes[0].plot(complexity_list, np.array(complexity_list) * baseline_slope + baseline_intercept,
                 'b--', linewidth=2, label=f'Fit: r={baseline_corr:.3f}')
    axes[0].set_xlabel('Scene Complexity', fontsize=12)
    axes[0].set_ylabel('Mean Entropy', fontsize=12)
    axes[0].set_title('Baseline: Complexity vs Entropy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(complexity_list, ensemble_entropy_list, alpha=0.5, label='Ensemble', s=30, color='coral')
    axes[1].plot(complexity_list, np.array(complexity_list) * ensemble_slope + ensemble_intercept,
                 'r--', linewidth=2, label=f'Fit: r={ensemble_corr:.3f}')
    axes[1].set_xlabel('Scene Complexity', fontsize=12)
    axes[1].set_ylabel('Mean Entropy', fontsize=12)
    axes[1].set_title('Ensemble: Complexity vs Entropy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complexity_scatter.png'), dpi=150)
    plt.close()
    
    # Save statistics
    stats_dict = {
        'baseline_correlation': float(baseline_corr),
        'baseline_p_value': float(baseline_p),
        'baseline_slope': float(baseline_slope),
        'ensemble_correlation': float(ensemble_corr),
        'ensemble_p_value': float(ensemble_p),
        'ensemble_slope': float(ensemble_slope),
    }
    
    with open(os.path.join(save_dir, 'complexity_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"\n✅ Test 4 Complete!")
    print(f"   Baseline: r={baseline_corr:.3f}, p={baseline_p:.4f}")
    print(f"   Ensemble: r={ensemble_corr:.3f}, p={ensemble_p:.4f}")
    print(f"   {'✓ Positive correlation' if ensemble_corr > 0 else '✗ Negative correlation'}")
    
    return stats_dict

def test_5_ensemble_disagreement(ensemble_model, val_loader, device, save_dir):
    """
    Test 5: Ensemble Disagreement Validation
    Verify that entropy captures model disagreement (ensemble-specific test)
    """
    print("\n" + "="*60)
    print("TEST 5: ENSEMBLE DISAGREEMENT VALIDATION")
    print("="*60)
    
    ensemble_model.eval()
    
    variance_list = []
    entropy_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Analyzing disagreement")):
            if i >= NUM_TEST_SCENES:
                break
            
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            imgs = imgs.to(device)
            rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device)
            post_rots, post_trans = post_rots.to(device), post_trans.to(device)
            
            # Ensemble predictions
            preds = ensemble_model(imgs, rots, trans, intrins, post_rots, post_trans)
            probs = torch.sigmoid(preds)  # (B, N_models, 1, H, W)
            
            # Model disagreement (variance)
            variance = probs.var(dim=1)[0, 0].cpu()
            
            # Entropy from mean prediction
            mean_prob = probs.mean(dim=1)[0, 0].cpu()
            entropy = compute_entropy(mean_prob)
            
            variance_list.extend(variance.flatten().numpy())
            entropy_list.extend(entropy.flatten().numpy())
    
    variance_array = np.array(variance_list)
    entropy_array = np.array(entropy_list)
    
    # Statistical analysis
    correlation, p_value = pearsonr(variance_array, entropy_array)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with density
    from scipy.stats import gaussian_kde
    
    # Sample for visualization (too many points otherwise)
    sample_size = min(10000, len(variance_array))
    idx = np.random.choice(len(variance_array), sample_size, replace=False)
    var_sample = variance_array[idx]
    ent_sample = entropy_array[idx]
    
    axes[0].hexbin(var_sample, ent_sample, gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[0].set_xlabel('Model Disagreement (Variance)', fontsize=12)
    axes[0].set_ylabel('Entropy', fontsize=12)
    axes[0].set_title(f'Disagreement vs Entropy (r={correlation:.3f})', fontsize=14, fontweight='bold')
    
    # Regression line
    slope, intercept = np.polyfit(variance_array, entropy_array, 1)
    x_line = np.linspace(variance_array.min(), variance_array.max(), 100)
    y_line = slope * x_line + intercept
    axes[0].plot(x_line, y_line, 'b--', linewidth=3, label='Linear fit')
    axes[0].legend()
    
    # Binned analysis
    n_bins = 10
    var_bins = np.percentile(variance_array, np.linspace(0, 100, n_bins + 1))
    bin_means_var = []
    bin_means_ent = []
    bin_stds_ent = []
    
    for i in range(n_bins):
        mask = (variance_array >= var_bins[i]) & (variance_array < var_bins[i+1])
        if mask.sum() > 0:
            bin_means_var.append(variance_array[mask].mean())
            bin_means_ent.append(entropy_array[mask].mean())
            bin_stds_ent.append(entropy_array[mask].std())
    
    axes[1].errorbar(bin_means_var, bin_means_ent, yerr=bin_stds_ent, fmt='o-', capsize=5, linewidth=2, markersize=8)
    axes[1].set_xlabel('Model Disagreement (Variance) - Binned', fontsize=12)
    axes[1].set_ylabel('Mean Entropy', fontsize=12)
    axes[1].set_title('Binned Disagreement vs Entropy', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_disagreement.png'), dpi=150)
    plt.close()
    
    # Save statistics
    stats_dict = {
        'correlation': float(correlation),
        'p_value': float(p_value),
        'slope': float(slope),
        'intercept': float(intercept),
        'variance_mean': float(variance_array.mean()),
        'variance_std': float(variance_array.std()),
        'entropy_mean': float(entropy_array.mean()),
        'entropy_std': float(entropy_array.std()),
    }
    
    with open(os.path.join(save_dir, 'ensemble_disagreement.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"\n✅ Test 5 Complete!")
    print(f"   Correlation: r={correlation:.3f}, p={p_value:.4e}")
    print(f"   {'✓ Strong positive correlation' if correlation > 0.7 else '⚠ Weak correlation'}")
    
    return stats_dict

# ==========================================
#                 MAIN
# ==========================================

def main():
    # Setup results directory based on model name
    RESULTS_DIR = setup_results_dir(ENSEMBLE_PATH)
    model_name = get_model_name_from_path(ENSEMBLE_PATH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print(f"📝 Testing model: {model_name}")
    print(f"📁 Results will be saved to: {RESULTS_DIR}")
    
    # Configuration
    grid_conf = {
        'xbound': [-50, 50, 0.5],
        'ybound': [-50, 50, 0.5],
        'zbound': [-10, 10, 20],
        'dbound': [4.0, 45.0, 1.0]
    }
    
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
    
    # Load models
    print("\n📦 Loading models...")
    
    # Baseline
    print(f"   Loading baseline from: {BASELINE_PATH}")
    baseline_model = BaselineLSS(grid_conf, data_aug_conf, outC=1).to(device)
    baseline_ckpt = torch.load(BASELINE_PATH, map_location=device)
    baseline_model.model.load_state_dict(baseline_ckpt, strict=False)
    baseline_model.eval()
    print("   ✓ Baseline loaded")
    
    # Ensemble
    print(f"   Loading ensemble from: {ENSEMBLE_PATH}")
    ensemble_model = EnsembleLSS(grid_conf, data_aug_conf, outC=1, num_models=5).to(device)
    ensemble_ckpt = torch.load(ENSEMBLE_PATH, map_location=device)
    ensemble_model.load_state_dict({k.replace('module.', ''): v for k, v in ensemble_ckpt['state_dict'].items()})
    ensemble_model.eval()
    print("   ✓ Ensemble loaded")
    
    # Load data
    print("\n📂 Loading validation data...")
    _, val_loader = compile_data('mini', DATAROOT, data_aug_conf, grid_conf,
                                  bsz=1, nworkers=4, parser_name='segmentationdata')
    print(f"   ✓ Loaded {len(val_loader)} validation scenes")
    
    # Run tests
    print("\n" + "="*60)
    print("🧪 STARTING ENTROPY VALIDATION TEST SUITE")
    print("="*60)
    
    all_stats = {}
    
    # Test 1: Darkness
    all_stats['test_1'] = test_1_darkness_sensitivity(baseline_model, ensemble_model, val_loader, device, RESULTS_DIR)
    
    # Test 2: Distance
    all_stats['test_2'] = test_2_distance_uncertainty(baseline_model, ensemble_model, val_loader, device, grid_conf, RESULTS_DIR)
    
    # Test 3: Occlusion
    all_stats['test_3'] = test_3_occlusion_response(baseline_model, ensemble_model, val_loader, device, RESULTS_DIR)
    
    # Test 4: Complexity
    all_stats['test_4'] = test_4_complexity_correlation(baseline_model, ensemble_model, val_loader, device, RESULTS_DIR)
    
    # Test 5: Disagreement (ensemble only)
    all_stats['test_5'] = test_5_ensemble_disagreement(ensemble_model, val_loader, device, RESULTS_DIR)
    
    # Generate summary report
    generate_summary_report(all_stats, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE!")
    print(f"📊 Results saved to: {RESULTS_DIR}")
    print(f"📊 Example images saved to: {os.path.join(RESULTS_DIR, 'examples')}")
    print("="*60)

def generate_summary_report(all_stats, save_dir):
    """Generate comprehensive summary report"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("         ENTROPY VALIDATION SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Test 1
    report_lines.append("TEST 1: DARKNESS SENSITIVITY")
    report_lines.append("-" * 80)
    t1 = all_stats['test_1']
    report_lines.append(f"  Baseline Correlation: r={t1['baseline_correlation']:.3f}, p={t1['baseline_p_value']:.4f}")
    report_lines.append(f"  Ensemble Correlation: r={t1['ensemble_correlation']:.3f}, p={t1['ensemble_p_value']:.4f}")
    winner = "ENSEMBLE" if abs(t1['ensemble_correlation']) > abs(t1['baseline_correlation']) else "BASELINE"
    report_lines.append(f"  Winner: {winner} ({'stronger negative correlation with brightness' if winner == 'ENSEMBLE' else ''})")
    report_lines.append("")
    
    # Test 2
    report_lines.append("TEST 2: DISTANCE-BASED UNCERTAINTY")
    report_lines.append("-" * 80)
    t2 = all_stats['test_2']
    report_lines.append(f"  Baseline ANOVA: F={t2['baseline_anova_f']:.2f}, p={t2['baseline_anova_p']:.4f}")
    report_lines.append(f"  Ensemble ANOVA: F={t2['ensemble_anova_f']:.2f}, p={t2['ensemble_anova_p']:.4f}")
    report_lines.append(f"  Ensemble Mean Entropy: Near={t2['ensemble_means'][0]:.4f}, Mid={t2['ensemble_means'][1]:.4f}, Far={t2['ensemble_means'][2]:.4f}")
    increasing = t2['ensemble_means'][2] > t2['ensemble_means'][0]
    report_lines.append(f"  Entropy increases with distance: {'YES ✓' if increasing else 'NO ✗'}")
    report_lines.append("")
    
    # Test 3
    report_lines.append("TEST 3: OCCLUSION RESPONSE")
    report_lines.append("-" * 80)
    t3 = all_stats['test_3']
    report_lines.append(f"  Baseline Entropy Change: Δ={t3['baseline_mean_delta']:.4f} ± {t3['baseline_std_delta']:.4f}")
    report_lines.append(f"  Ensemble Entropy Change: Δ={t3['ensemble_mean_delta']:.4f} ± {t3['ensemble_std_delta']:.4f}")
    report_lines.append(f"  Ensemble p-value: {t3['ensemble_ttest_pvalue']:.4f}")
    significant = t3['ensemble_ttest_pvalue'] < 0.05 and t3['ensemble_mean_delta'] > 0
    report_lines.append(f"  Significant positive increase: {'YES ✓' if significant else 'NO ✗'}")
    report_lines.append("")
    
    # Test 4
    report_lines.append("TEST 4: SCENE COMPLEXITY CORRELATION")
    report_lines.append("-" * 80)
    t4 = all_stats['test_4']
    report_lines.append(f"  Baseline Correlation: r={t4['baseline_correlation']:.3f}, p={t4['baseline_p_value']:.4f}")
    report_lines.append(f"  Ensemble Correlation: r={t4['ensemble_correlation']:.3f}, p={t4['ensemble_p_value']:.4f}")
    winner = "ENSEMBLE" if abs(t4['ensemble_correlation']) > abs(t4['baseline_correlation']) else "BASELINE"
    report_lines.append(f"  Winner: {winner}")
    report_lines.append("")
    
    # Test 5
    report_lines.append("TEST 5: ENSEMBLE DISAGREEMENT VALIDATION")
    report_lines.append("-" * 80)
    t5 = all_stats['test_5']
    report_lines.append(f"  Variance-Entropy Correlation: r={t5['correlation']:.3f}, p={t5['p_value']:.4e}")
    strong = t5['correlation'] > 0.7
    report_lines.append(f"  Strong positive correlation: {'YES ✓' if strong else 'MODERATE ⚠'}")
    report_lines.append("")
    
    # Overall Assessment
    report_lines.append("="*80)
    report_lines.append("OVERALL ASSESSMENT")
    report_lines.append("="*80)
    
    scores = {
        'darkness': 1 if abs(t1['ensemble_correlation']) > abs(t1['baseline_correlation']) else 0,
        'distance': 1 if increasing and t2['ensemble_anova_p'] < 0.05 else 0,
        'occlusion': 1 if significant else 0,
        'complexity': 1 if t4['ensemble_correlation'] > 0.3 else 0,
        'disagreement': 1 if strong else 0
    }
    
    total_score = sum(scores.values())
    
    report_lines.append(f"Tests Passed: {total_score}/5")
    report_lines.append("")
    report_lines.append(f"✓ Darkness Sensitivity: {'PASS' if scores['darkness'] else 'FAIL'}")
    report_lines.append(f"✓ Distance Uncertainty: {'PASS' if scores['distance'] else 'FAIL'}")
    report_lines.append(f"✓ Occlusion Response: {'PASS' if scores['occlusion'] else 'FAIL'}")
    report_lines.append(f"✓ Complexity Correlation: {'PASS' if scores['complexity'] else 'FAIL'}")
    report_lines.append(f"✓ Ensemble Disagreement: {'PASS' if scores['disagreement'] else 'FAIL'}")
    report_lines.append("")
    
    if total_score >= 4:
        recommendation = "EXCELLENT - Entropy is highly meaningful for radar guidance. Proceed to simulation."
    elif total_score >= 3:
        recommendation = "GOOD - Entropy shows meaningful patterns. Consider retraining with differential LR for improvement."
    else:
        recommendation = "NEEDS IMPROVEMENT - Consider retraining with differential LR and more data."
    
    report_lines.append(f"RECOMMENDATION: {recommendation}")
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(os.path.join(save_dir, 'summary_report.txt'), 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)

if __name__ == "__main__":
    main()

