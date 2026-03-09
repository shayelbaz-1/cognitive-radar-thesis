import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# --- CONFIG ---
CKPT_PATH = 'checkpoints/runs/mini/hybrid_entropy_new/2025-12-27_22-09-42/checkpoints/model_best.pth'
DATAROOT = './data/'
# --------------

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

sys.path.append(os.path.join(os.getcwd(), 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

def denormalize_img(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

class FullDeepEnsembleLSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5):
        super().__init__()
        self.models = nn.ModuleList()
        # אנחנו צריכים רק מודל אחד כדי לבדוק את העומק הפנימי שלו
        m = LiftSplatShoot(grid_conf, data_aug_conf, outC=outC)
        self.models.append(m)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # אנחנו לא מריצים Forward רגיל, אלא ניגשים ישר ל-CamEncode
        # x: (B, N, C, H, W) -> (B*N, C, H, W)
        B, N, C, H, W = x.shape
        x_flat = x.view(B*N, C, H, W)
        
        # שלב ה-LIFT: קבלת העומק והפיצ'רים
        # הפונקציה get_depth_dist בד"כ מחזירה את ה-Softmax על העומק
        # אנחנו נצטרך לגשת לתוך camencode
        return self.models[0].camencode(x_flat)

def apply_patch_shuffle(img_tensor):
    # אותו שפל בדיוק
    out = img_tensor.clone()
    B, N, C, H, W = out.shape
    h_step, w_step = H // 4, W // 4
    blocks = []
    for h in range(4):
        for w in range(4):
            blocks.append(out[:, :, :, h*h_step:(h+1)*h_step, w*w_step:(w+1)*w_step])
    idx = torch.randperm(16)
    shuffled_blocks = [blocks[i] for i in idx]
    new_out = torch.zeros_like(out)
    k = 0
    for h in range(4):
        for w in range(4):
            new_out[:, :, :, h*h_step:(h+1)*h_step, w*w_step:(w+1)*w_step] = shuffled_blocks[k]
            k += 1
    return new_out

def calc_depth_entropy(depth_logits):
    """
    depth_logits: (B*N, D, H_down, W_down) - D channels represents depth bins
    """
    # המרת לוגיטים להסתברויות (Softmax על מימד העומק)
    probs = torch.softmax(depth_logits, dim=1)
    
    # חישוב אנטרופיה על מימד העומק (Dimension 1)
    eps = 1e-7
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1) # (B*N, H_down, W_down)
    
    return entropy

def main():
    grid_conf = {'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5], 'zbound': [-10, 10, 20], 'dbound': [4.0, 45.0, 1.0]}
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {'resize_lim': (0.193, 0.193), 'final_dim': (128, 352), 'rot_lim': (0.0, 0.0), 'H': 900, 'W': 1600, 'bot_pct_lim': (0.0, 0.0), 'cams': cams, 'Ncams': 6, 'rand_flip': False}

    device = torch.device('cuda')
    model = FullDeepEnsembleLSS(grid_conf, data_aug_conf, num_models=1).to(device)
    
    print("Loading weights...")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    # טעינה למודל הבודד
    state_dict = {k.replace('module.', '').replace('models.0.', ''): v for k, v in ckpt['state_dict'].items() if 'models.0' in k or 'module' not in k}
    model.models[0].load_state_dict(state_dict, strict=False)
    model.eval()

    loader, _ = compile_data('mini', DATAROOT, data_aug_conf, grid_conf, bsz=1, nworkers=0, parser_name='segmentationdata')
    
    # שליפת באץ' ראשון
    batch = next(iter(loader))
    imgs, rots, trans, intrins, post_rots, post_trans, _ = batch
    imgs = imgs.to(device)

    print("Checking DEPTH Entropy...")

    # 1. Clean Depth Entropy
    with torch.no_grad():
        # camencode מחזיר (depth_logits, features)
        depth_logits_clean, _ = model(imgs, None, None, None, None, None)
        ent_clean = calc_depth_entropy(depth_logits_clean) # (6, H, W)
        avg_ent_clean = ent_clean.mean().item()

    # 2. Shuffled Depth Entropy
    imgs_shuff = apply_patch_shuffle(imgs)
    with torch.no_grad():
        depth_logits_shuff, _ = model(imgs_shuff, None, None, None, None, None)
        ent_shuff = calc_depth_entropy(depth_logits_shuff)
        avg_ent_shuff = ent_shuff.mean().item()

    print("\n" + "="*40)
    print(f"Depth Entropy CLEAN:    {avg_ent_clean:.4f}")
    print(f"Depth Entropy SHUFFLED: {avg_ent_shuff:.4f}")
    print("="*40)

    # ויזואליזציה (מצלמה קדמית בלבד)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Clean
    axes[0, 0].imshow(denormalize_img(imgs[0, 1]))
    axes[0, 0].set_title("Input Clean")
    im1 = axes[0, 1].imshow(ent_clean[1].cpu().numpy(), cmap='magma')
    axes[0, 1].set_title(f"Depth Entropy (Clean)\nAvg: {ent_clean[1].mean():.4f}")
    plt.colorbar(im1, ax=axes[0, 1])

    # Shuffled
    axes[1, 0].imshow(denormalize_img(imgs_shuff[0, 1]))
    axes[1, 0].set_title("Input Shuffled")
    im2 = axes[1, 1].imshow(ent_shuff[1].cpu().numpy(), cmap='magma')
    axes[1, 1].set_title(f"Depth Entropy (Shuffled)\nAvg: {ent_shuff[1].mean():.4f}")
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('depth_entropy_check.png')
    print("✅ Saved 'depth_entropy_check.png'")

if __name__ == "__main__":
    main()