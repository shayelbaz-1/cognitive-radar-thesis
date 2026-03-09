import torch
import torch.nn as nn
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
import PIL.Image

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

sys.path.append(os.path.join(os.getcwd(), 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

# ==========================================
#      MODEL DEFINITIONS
# ==========================================
class FullDeepEnsembleLSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList()
        for i in range(num_models):
            m = LiftSplatShoot(grid_conf, data_aug_conf, outC=outC)
            self.models.append(m)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        outputs = []
        for model in self.models:
            res = model(x, rots, trans, intrins, post_rots, post_trans)
            outputs.append(res)
        return torch.stack(outputs, dim=1)

# ==========================================
#      METRICS
# ==========================================
def binary_entropy(p):
    eps = 1e-7
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)

def calc_metrics(preds):
    probs = torch.sigmoid(preds)
    mean_prob = probs.mean(dim=1)
    
    total_entropy = binary_entropy(mean_prob)
    aleatoric = binary_entropy(probs).mean(dim=1)
    mutual_info = total_entropy - aleatoric
    std_dev = probs.std(dim=1)
    
    return mean_prob, mutual_info, std_dev

# ==========================================
#      VISUALIZATION HELPER
# ==========================================
def add_ego_marker(ax):
    """
    מצייר משולש אדום במרכז המפה (הרכב).
    """
    center_x, center_y = 100, 100
    triangle_points = [
        [center_x, center_y + 6],       
        [center_x - 3, center_y - 4],   
        [center_x + 3, center_y - 4]    
    ]
    poly = patches.Polygon(triangle_points, closed=True, facecolor='red', edgecolor=None, zorder=10)
    ax.add_patch(poly)
    poly_border = patches.Polygon(triangle_points, closed=True, facecolor='none', edgecolor='white', linewidth=1, zorder=11)
    ax.add_patch(poly_border)

# ==========================================
#      VISUALIZATION (PRO LAYOUT)
# ==========================================
def save_dashboard(imgs, gt, orig_pred, mean_pred, std, mi, output_path, sample_idx):
    # --- 1. הכנת הנתונים ---
    cams = imgs[0].cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std_norm = np.array([0.229, 0.224, 0.225])
    cams = (cams * std_norm + mean)
    cams = np.clip(cams, 0, 1)

    # === תיקון היפוך ימין/שמאל ===
    # ב-LSS לפעמים ציר ה-X בתמונה הוא הפוך לציר ה-Y במטריצה.
    # [::-1] הופך את הציר האופקי (Horizontal Flip)
    
    gt_map = gt[0, 0].cpu().numpy()[:, ::-1] 
    orig_prob = torch.sigmoid(orig_pred[0, 0]).cpu().numpy()[:, ::-1]
    ens_mean = mean_pred[0, 0].cpu().numpy()[:, ::-1]
    ens_std = std[0, 0].cpu().numpy()[:, ::-1]
    ens_mi = mi[0, 0].cpu().numpy()[:, ::-1]

    # --- 2. הגדרת הלייאוט ---
    fig = plt.figure(figsize=(25, 16))
    outer_grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.6], hspace=0.25)

    # --- חלק עליון: מצלמות ---
    inner_grid_cams = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_grid[0:2], wspace=0.1, hspace=0.1)
    cam_names = ['Front Left', 'Front', 'Front Right', 'Back Left', 'Back', 'Back Right']
    
    for i in range(6):
        ax = plt.Subplot(fig, inner_grid_cams[i])
        ax.imshow(cams[i])
        ax.axis('off')
        ax.set_title(cam_names[i], fontsize=14)
        fig.add_subplot(ax)

    # --- חלק תחתון: מפות BEV ---
    inner_grid_bev = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_grid[2], wspace=0.35)

    maps_config = [
        {'data': gt_map, 'title': 'Ground Truth', 'cmap': 'gray', 'clim': (0, 1), 'cbar': False},
        {'data': orig_prob, 'title': 'Original Model', 'cmap': 'magma', 'clim': (0, 1), 'cbar': True},
        {'data': ens_mean, 'title': 'Ensemble Mean', 'cmap': 'magma', 'clim': (0, 1), 'cbar': True},
        {'data': ens_std, 'title': 'Disagreement (Std)', 'cmap': 'jet', 'clim': (0, np.max(ens_std) + 1e-6), 'cbar': True},
        {'data': ens_mi, 'title': 'Uncertainty (MI)', 'cmap': 'jet', 'clim': (0, np.max(ens_mi) + 1e-6), 'cbar': True},
    ]

    for i, config in enumerate(maps_config):
        ax = plt.Subplot(fig, inner_grid_bev[i])
        fig.add_subplot(ax)
        
        im = ax.imshow(config['data'], cmap=config['cmap'], 
                       vmin=config['clim'][0], vmax=config['clim'][1], 
                       origin='lower')
        
        ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks([])
        ax.set_yticks([])

        add_ego_marker(ax)

        if config['cbar']:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad=0.1)
            plt.colorbar(im, cax=cax, orientation='vertical')

    plt.savefig(os.path.join(output_path, f"vis_pro_{sample_idx:04d}.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)

# ==========================================
#             MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_ckpt', type=str, required=True, help='Path to ensemble model_best.pth')
    parser.add_argument('--original_ckpt', type=str, default=None, help='Path to original single model weights')
    parser.add_argument('--dataroot', type=str, default='./data/')
    parser.add_argument('--version', type=str, default='mini')
    parser.add_argument('--dropout_test', action='store_true', help='Apply noise to Front-Left camera?')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'final_visualization_pro'
    os.makedirs(output_dir, exist_ok=True)

    grid_conf = {
        'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5],
        'zbound': [-10, 10, 20], 'dbound': [4.0, 45.0, 1.0],
    }
    
    # === תיקון קריטי: rand_flip חייב להיות False בולידציה! ===
    data_aug_conf = {
        'resize_lim': (0.193, 0.225), 'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600,
        'bot_pct_lim': (0.0, 0.0), 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6, 
        'rand_flip': False  # <--- השינוי החשוב
    }

    print("🔄 Loading Validation Data...")
    _, val_loader = compile_data(
        args.version, args.dataroot, data_aug_conf, grid_conf, 
        bsz=1, nworkers=4, parser_name='segmentationdata'
    )

    print(f"🧠 Loading ENSEMBLE from {args.ensemble_ckpt}...")
    model_ens = FullDeepEnsembleLSS(grid_conf, data_aug_conf, outC=1, num_models=5)
    ckpt = torch.load(args.ensemble_ckpt, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model_ens.load_state_dict(new_state_dict)
    model_ens.to(device).eval()

    print(f"🧠 Loading ORIGINAL Single Model...")
    model_orig = LiftSplatShoot(grid_conf, data_aug_conf, outC=1)
    if args.original_ckpt:
        print(f"   -> Loading weights from {args.original_ckpt}")
        orig_ckpt = torch.load(args.original_ckpt, map_location=device)
        sd = orig_ckpt['state_dict'] if 'state_dict' in orig_ckpt else orig_ckpt
        clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model_orig.load_state_dict(clean_sd)
    else:
        print("   -> ⚠️ No checkpoint provided for original model! Using ImageNet (Untrained).")
    model_orig.to(device).eval()

    print("🚀 Starting Visualization Loop...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            
            imgs = imgs.to(device)
            binimgs = binimgs.to(device)
            rots, trans, intrins = rots.to(device), trans.to(device), intrins.to(device)
            post_rots, post_trans = post_rots.to(device), post_trans.to(device)

            if args.dropout_test:
                # הזרקת רעש למצלמה מס' 0 (Front Left)
                noise = torch.randn_like(imgs[:, 0, :, :, :]) * 2.0 + 3.0
                imgs[:, 0, :, :, :] = noise

            preds_ens = model_ens(imgs, rots, trans, intrins, post_rots, post_trans)
            pred_orig = model_orig(imgs, rots, trans, intrins, post_rots, post_trans)

            mean_pred, mi, std = calc_metrics(preds_ens)

            save_dashboard(imgs, binimgs, pred_orig, mean_pred, std, mi, output_dir, i)
            
            if i >= 15: break

    print(f"✅ Visualization saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()