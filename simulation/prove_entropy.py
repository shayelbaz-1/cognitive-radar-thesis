"""
Entropy Validation Suite
========================

This script is a dedicated tool for validating the efficacy of Shannon Entropy
as an active sensing metric. It generates scientific proofs required for thesis defense.

Outputs:
1. Sparsification Plot (Entropy vs Random vs Oracle) - The "Gold Standard" proof.
2. Calibration Plot (Entropy vs Error) - Shows correlation.
3. Property Analysis (Missed Detections, Edge Precision).

Usage:
    python prove_entropy.py --model_path path/to/model.pth --num_scenes 50
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion

# Setup paths (same as your existing simulation)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

# Import your existing config and utils
from config import RadarConfig, SimulationConfig
from information_theory import compute_entropy
from scene_conditions import get_scene_groups_from_dataset


class EntropyValidator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.radar_config = RadarConfig()
        self.sim_config = SimulationConfig()
        
        print(f"🔧 Loading model from: {model_path}")
        self._load_model(model_path)
        self._load_data()
        
        # Directory for proofs
        self.output_dir = os.path.join(os.path.dirname(__file__), 'entropy_proofs')
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self, model_path):
        # Using the same Ensemble/LSS architecture from your simulation
        data_aug_conf = {
            'resize_lim': (0.193, 0.193), 'final_dim': (128, 352),
            'rot_lim': (0.0, 0.0), 'H': 900, 'W': 1600,
            'bot_pct_lim': (0.0, 0.0), 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
                                                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': 6, 'rand_flip': False
        }
        # Wrapper for Ensemble loading
        class EnsembleLSS(torch.nn.Module):
            def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5):
                super().__init__()
                self.models = torch.nn.ModuleList([LiftSplatShoot(grid_conf, data_aug_conf, outC=outC) for _ in range(num_models)])
            def forward(self, x, rots, trans, intrins, post_rots, post_trans):
                return torch.stack([m(x, rots, trans, intrins, post_rots, post_trans) for m in self.models], dim=1)

        self.model = EnsembleLSS(self.sim_config.grid_conf, data_aug_conf).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
        self.model.eval()

    def _load_data(self):
        print("📂 Loading validation data...")
        data_path = os.path.join(parent_dir, 'data_nuscenes')
        # Re-using data loader logic
        data_aug_conf = {'resize_lim': (0.193, 0.193), 'final_dim': (128, 352), 'rot_lim': (0.0, 0.0), 'H': 900, 'W': 1600, 'bot_pct_lim': (0.0, 0.0), 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'], 'Ncams': 6, 'rand_flip': False}
        _, self.val_loader = compile_data('trainval', data_path, data_aug_conf, self.sim_config.grid_conf, bsz=1, nworkers=4, parser_name='segmentationdata')
        print(f"   ✓ Loaded {len(self.val_loader)} scenes")
    def collect_global_statistics(self, num_scenes=50):
        """
        Runs the model on N scenes and collects PIXEL-LEVEL statistics.
        Returns flattened arrays of everything needed for the proofs,
        both globally and per condition (DAY/NIGHT/RAINY/etc).
        
        Returns:
            (global_data, by_condition, global_scene_count, by_condition_scene_count)
        """
        print(f"📊 Collecting statistics from {num_scenes} scenes...")
        
        # Storage containers (global)
        all_entropy = []
        all_error = []
        all_gt = []
        all_pred = []
        all_is_edge = []
        
        # Storage containers (per-condition)
        by_condition = {}
        
        # Scene count tracking
        global_scene_count = 0
        by_condition_scene_count = {}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, total=num_scenes)):
                if i >= num_scenes: break
                
                imgs, rots, trans, intrins, post_rots, post_trans, gt_binimg = batch
                imgs = imgs.to(self.device)
                rots, trans, intrins = rots.to(self.device), trans.to(self.device), intrins.to(self.device)
                post_rots, post_trans = post_rots.to(self.device), post_trans.to(self.device)
                
                # Model Inference
                preds = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
                probs = torch.sigmoid(preds)
                mean_prob = probs.mean(dim=1)[0, 0].cpu().numpy()
                mean_prob = np.fliplr(mean_prob) # Correct orientation
                gt = np.fliplr(gt_binimg[0, 0].numpy())
                
                # Compute metrics maps
                entropy_map = compute_entropy(mean_prob)
                error_map = np.abs(mean_prob - gt)
                
                # Compute Edges (for Property Analysis)
                gt_bool = gt > 0.5
                edges = binary_dilation(gt_bool) ^ binary_erosion(gt_bool)
                
                # Flatten
                entropy_flat = entropy_map.flatten()
                error_flat = error_map.flatten()
                gt_flat = gt.flatten()
                pred_flat = mean_prob.flatten()
                edge_flat = edges.flatten()
                
                # Append to global arrays
                all_entropy.append(entropy_flat)
                all_error.append(error_flat)
                all_gt.append(gt_flat)
                all_pred.append(pred_flat)
                all_is_edge.append(edge_flat)
                
                # Increment global scene count
                global_scene_count += 1
                
                # Classify scene and append to per-condition buckets
                scene_groups = get_scene_groups_from_dataset(self.val_loader.dataset, i)
                for group in scene_groups:
                    if group not in by_condition:
                        by_condition[group] = {
                            'entropy': [], 'error': [], 'gt': [], 
                            'pred': [], 'is_edge': []
                        }
                    by_condition[group]['entropy'].append(entropy_flat)
                    by_condition[group]['error'].append(error_flat)
                    by_condition[group]['gt'].append(gt_flat)
                    by_condition[group]['pred'].append(pred_flat)
                    by_condition[group]['is_edge'].append(edge_flat)
                    
                    # Increment per-condition scene count
                    if group not in by_condition_scene_count:
                        by_condition_scene_count[group] = 0
                    by_condition_scene_count[group] += 1
        
        # Concatenate global arrays
        print("   ✓ Concatenating global data...")
        global_data = {
            'entropy': np.concatenate(all_entropy),
            'error': np.concatenate(all_error),
            'gt': np.concatenate(all_gt),
            'pred': np.concatenate(all_pred),
            'is_edge': np.concatenate(all_is_edge).astype(bool)
        }
        
        # Concatenate per-condition arrays
        print("   ✓ Concatenating per-condition data...")
        for condition in by_condition:
            by_condition[condition] = {
                'entropy': np.concatenate(by_condition[condition]['entropy']),
                'error': np.concatenate(by_condition[condition]['error']),
                'gt': np.concatenate(by_condition[condition]['gt']),
                'pred': np.concatenate(by_condition[condition]['pred']),
                'is_edge': np.concatenate(by_condition[condition]['is_edge']).astype(bool)
            }
        
        print(f"   ✓ Collected {global_scene_count} scenes total")
        print(f"   ✓ Per-condition counts: {by_condition_scene_count}")
        
        return global_data, by_condition, global_scene_count, by_condition_scene_count

    def plot_sparsification_oracle(self, data, suffix='', condition_label='GLOBAL', scene_count=None):
        """
        PROOF 1: Sparsification Plot (Fixed: Float64 Precision & Normalization)
        
        Args:
            data: Dict with entropy, error, gt, pred, is_edge arrays
            suffix: Optional suffix for filename (e.g., '_DAY_SCENES')
            condition_label: Label for the scene condition (e.g., 'GLOBAL', 'DAY_SCENES')
            scene_count: Number of scenes used in this analysis
        """
        print("   📈 Generating Sparsification Plot...")
        
        # המרה ל-Float64 קריטית כשסוכמים מיליוני פיקסלים!
        entropy_vals = data['entropy'].astype(np.float64)
        error_vals = data['error'].astype(np.float64)
        N = len(error_vals)
        
        # 1. Oracle Sort
        idx_oracle = np.argsort(error_vals)[::-1]
        sorted_oracle = error_vals[idx_oracle]
        
        # 2. Entropy Sort
        idx_entropy = np.argsort(entropy_vals)[::-1]
        sorted_entropy = error_vals[idx_entropy]
        
        # 3. Random Sort
        idx_random = np.random.permutation(N)
        sorted_random = error_vals[idx_random]
        
        # --- CRITICAL FIX: Robust Normalization ---
        # במקום להשתמש ב-sum() חיצוני, נשתמש בערך האחרון של ה-cumsum
        # זה מבטיח מתמטית שהגרף יסתיים ב-0 בדיוק, גם אם יש שגיאות עיגול זעירות.
        
        cumsum_oracle = np.cumsum(sorted_oracle)
        total_error = cumsum_oracle[-1] # This ensures the last point is exactly 0
        
        rem_oracle = (total_error - cumsum_oracle) / total_error
        rem_entropy = (total_error - np.cumsum(sorted_entropy)) / total_error
        rem_random = (total_error - np.cumsum(sorted_random)) / total_error
        
        # Smart downsampling including the last point
        step = max(1, N // 1000)
        indices = np.arange(0, N, step)
        if indices[-1] != N - 1:
            indices = np.append(indices, N - 1)
            
        x_axis = np.linspace(0, 100, N)
        x_plot = x_axis[indices]
        
        plt.figure(figsize=(10, 7))
        plt.plot(x_plot, rem_random[indices], 'k--', linewidth=2, label='Random (Lower Bound)', alpha=0.7)
        plt.plot(x_plot, rem_oracle[indices], 'b:', linewidth=3, label='Oracle / Optimal (Upper Bound)')
        plt.plot(x_plot, rem_entropy[indices], 'g-', linewidth=3, label='Entropy-Guided (Ours)')
        
        # Fill the "Optimality Gap"
        plt.fill_between(x_plot, rem_oracle[indices], rem_entropy[indices], color='green', alpha=0.1, label='Optimality Gap')
        
        plt.xlabel('Pixels Corrected (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Remaining Total Error (Normalized)', fontsize=12, fontweight='bold')
        
        # Build title with condition and scene count
        title = 'Sparsification Plot: Efficiency of Uncertainty Metric'
        if scene_count is not None:
            title += f'\nCondition: {condition_label} | Scenes: {scene_count}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.xlim([0, 100])
        plt.ylim([0, 1])
        
        # Add AUSE Score text
        diff = rem_entropy[indices] - rem_oracle[indices]
        ause = np.trapz(diff, x_plot/100.0) # מנרמלים את ציר ה-X ל-0..1
        
        # הוספת הטקסט לגרף
        plt.text(50, 0.6, f'AUSE: {ause:.4f}\n(Lower is better)', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), 
                 fontsize=12, fontweight='bold')
        
        filename = f'proof_1_sparsification_oracle{suffix}.png'
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"      Saved: {save_path}")
        print(f"      Saved: {save_path}")
        
        
    def plot_calibration(self, data, suffix='', condition_label='GLOBAL', scene_count=None):
        """
        PROOF 2: Reliability Diagram & Error Correlation
        Shows that High Entropy correlates with High Error.
        
        Args:
            data: Dict with entropy, error, gt, pred, is_edge arrays
            suffix: Optional suffix for filename (e.g., '_DAY_SCENES')
            condition_label: Label for the scene condition (e.g., 'GLOBAL', 'DAY_SCENES')
            scene_count: Number of scenes used in this analysis
        """
        print("   📈 Generating Calibration Plots...")
        
        entropy_vals = data['entropy']
        error_vals = data['error']
        
        # Divide into bins
        bins = np.linspace(0, 1.0, 21) # 5% bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_errors = []
        
        for i in range(len(bins)-1):
            mask = (entropy_vals >= bins[i]) & (entropy_vals < bins[i+1])
            if mask.sum() > 0:
                mean_errors.append(error_vals[mask].mean())
            else:
                mean_errors.append(0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, mean_errors, 'o-', color='darkorange', linewidth=2)
        plt.xlabel('Uncertainty (Entropy)', fontsize=12)
        plt.ylabel('Observed Error (MAE)', fontsize=12)
        
        # Build title with condition and scene count
        title = 'Calibration: Uncertainty vs. Error Correlation'
        if scene_count is not None:
            title += f'\nCondition: {condition_label} | Scenes: {scene_count}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.grid(True, linestyle='--')
        
        # Add correlation text
        corr = np.corrcoef(entropy_vals, error_vals)[0, 1]
        plt.text(0.05, 0.9, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='wheat', alpha=0.5))
        
        filename = f'proof_2_calibration{suffix}.png'
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"      Saved: {save_path}")

    def plot_properties(self, data, suffix='', condition_label='GLOBAL', scene_count=None):
        """
        PROOF 3: Full Confusion Matrix Analysis (Clean Version)
        
        Args:
            data: Dict with entropy, error, gt, pred, is_edge arrays
            suffix: Optional suffix for filename (e.g., '_DAY_SCENES')
            condition_label: Label for the scene condition (e.g., 'GLOBAL', 'DAY_SCENES')
            scene_count: Number of scenes used in this analysis
        """
        print("   📈 Generating Full Confusion Matrix Analysis...")
        
        entropy = data['entropy']
        gt = data['gt'] > 0.5
        pred = data['pred'] > 0.5 
        
        # --- 1. Define All 4 Groups ---
        mask_tn = (~gt) & (~pred) # Background
        mask_tp = gt & pred       # Correct Detection
        mask_fn = gt & (~pred)    # Missed Car
        mask_fp = (~gt) & pred    # Ghost Object
        
        # --- 2. Calculate Means ---
        def safe_mean(mask):
            return entropy[mask].mean() if mask.sum() > 0 else 0

        ent_tn = safe_mean(mask_tn)
        ent_tp = safe_mean(mask_tp)
        ent_fn = safe_mean(mask_fn)
        ent_fp = safe_mean(mask_fp)
        
        # --- 3. Plotting ---
        plt.figure(figsize=(10, 7))
        
        # Define categories and colors
        categories = ['True Neg\n(Background)', 'True Pos\n(Correct)', 'False Neg\n(Missed)', 'False Pos\n(Ghost)']
        values = [ent_tn, ent_tp, ent_fn, ent_fp]
        colors = ['#bdc3c7', '#27ae60', '#c0392b', '#8e44ad'] # Clean Flat UI colors
        
        # Create bars
        bars = plt.bar(categories, values, color=colors, alpha=0.9, edgecolor='black', width=0.6)
        
        plt.ylabel('Mean Uncertainty (Entropy)', fontsize=12, fontweight='bold')
        
        # Build title with condition and scene count
        title = 'Uncertainty Distribution by Error Type'
        if scene_count is not None:
            title += f'\nCondition: {condition_label} | Scenes: {scene_count}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add values on top (Clean numbers)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # --- 4. Explanatory Legend (Scientific, Top Left) ---
        # info_text = (
        #     r"$\bf{Interpretation:}$" + "\n"
        #     f"• TN (Gray): Low Uncertainty (Safe to ignore)\n"
        #     f"• TP (Green): Medium (Refinement needed)\n"
        #     f"• FN (Red): $\\bf{{High}}$ (Radar detects miss)\n"
        #     f"• FP (Purple): $\\bf{{Very\ High}}$ (Radar removes ghost)"
        # )
        
        # plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
        #          fontsize=11, verticalalignment='top',
        #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))

        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.ylim(0, max(values) * 1.2) # Give some headroom
        
        filename = f'proof_3_full_confusion_matrix{suffix}.png'
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"      Saved: {save_path}")

    def run(self, num_scenes=50):
        print("="*60)
        print("🧪 RUNNING ENTROPY PROOFS SUITE")
        print("="*60)
        
        global_data, by_condition, global_scene_count, by_condition_scene_count = self.collect_global_statistics(num_scenes)
        
        # Generate global proofs (existing behavior)
        print("\n📊 Generating global proofs...")
        self.plot_sparsification_oracle(global_data, suffix='', condition_label='GLOBAL', scene_count=global_scene_count)
        self.plot_calibration(global_data, suffix='', condition_label='GLOBAL', scene_count=global_scene_count)
        self.plot_properties(global_data, suffix='', condition_label='GLOBAL', scene_count=global_scene_count)
        
        # Generate per-condition proofs (new additive feature)
        print("\n📊 Generating per-condition proofs...")
        for condition, cond_data in by_condition.items():
            count = by_condition_scene_count[condition]
            print(f"   Processing {condition} ({count} scenes)...")
            suffix = f'_{condition}'
            self.plot_sparsification_oracle(cond_data, suffix=suffix, condition_label=condition, scene_count=count)
            self.plot_calibration(cond_data, suffix=suffix, condition_label=condition, scene_count=count)
            self.plot_properties(cond_data, suffix=suffix, condition_label=condition, scene_count=count)
        
        print("\n✅ All proofs generated in 'entropy_proofs/' folder.")
        print(f"   • Global proofs: proof_1_*.png, proof_2_*.png, proof_3_*.png")
        print(f"   • Per-condition: proof_*_DAY_SCENES.png, proof_*_NIGHT_SCENES.png, etc.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/shayelbaz/repos/checkpoints/runs/trainval/full_trainval_5heads_from_scratch/2026-01-04_15-23-47/checkpoints/model_best.pth', help='Path to .pth checkpoint')
    parser.add_argument('--num_scenes', type=int, default=50, help='Number of scenes to analyze')
    args = parser.parse_args()
    
    validator = EntropyValidator(args.model_path)
    validator.run(args.num_scenes)