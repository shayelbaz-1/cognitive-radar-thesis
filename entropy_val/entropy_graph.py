import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from sklearn.metrics import roc_auc_score, roc_curve

# --- CONFIG ---
CKPT_PATH = 'checkpoints/runs/mini/hybrid_entropy_new/2025-12-27_22-09-42/checkpoints/model_best.pth'
DATAROOT = './data/'
NUM_BATCHES = 20
# --------------

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

sys.path.append(os.path.join(os.getcwd(), 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

class FullDeepEnsembleLSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5):
        super().__init__()
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

# === NEW CORRUPTION: PATCH SHUFFLE ===
def apply_patch_shuffle(img_tensor, patch_size=32):
    """
    מפרק את התמונה לריבועים ומערבב אותם.
    זה משמר את הסטטיסטיקה (Mean/Std) אבל הורס את הסמנטיקה.
    המודל יראה 'טקסטורה של כביש/רכב' אבל לא יבין מה קורה -> אנטרופיה גבוהה.
    """
    # img_tensor: (B, N, C, H, W)
    out = img_tensor.clone()
    B, N, C, H, W = out.shape
    
    # נעבוד על כל תמונה בנפרד (קצת איטי אבל בטוח)
    # נשטח את הממדים הראשונים כדי לעבוד יעיל
    flat_imgs = out.view(-1, C, H, W) # (B*N, C, H, W)
    
    # חיתוך לפאצ'ים
    # unfold יוצר חלונות: (B*N, C, H_patches, W_patches, patch_h, patch_w)
    patches = flat_imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    bn, c, h_p, w_p, ph, pw = patches.shape
    
    # שיטוח הפאצ'ים לרשימה
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(bn, h_p * w_p, c, ph, pw)
    
    # ערבוב
    for i in range(bn):
        idx = torch.randperm(patches.shape[1])
        patches[i] = patches[i][idx]
        
    # הרכבה מחדש (Fold) - זה מסובך ב-PyTorch, נעשה הדבקה ידנית פשוטה יותר
    # כדי לא להסתבך עם fold, נשתמש בגישה פשוטה של החלפת בלוקים
    
    # גישה ב': פשוט נחליף קוביות רנדומלית
    # נחלק ל-4 רצועות רוחב ו-4 רצועות גובה (16 קוביות) ונערבב
    h_step = H // 4
    w_step = W // 4
    
    blocks = []
    for h in range(4):
        for w in range(4):
            blocks.append(out[:, :, :, h*h_step:(h+1)*h_step, w*w_step:(w+1)*w_step])
            
    # ערבוב האינדקסים
    idx = torch.randperm(16)
    shuffled_blocks = [blocks[i] for i in idx]
    
    # הרכבה מחדש
    new_out = torch.zeros_like(out)
    k = 0
    for h in range(4):
        for w in range(4):
            new_out[:, :, :, h*h_step:(h+1)*h_step, w*w_step:(w+1)*w_step] = shuffled_blocks[k]
            k += 1
            
    return new_out

def calc_entropy_score(preds):
    probs = torch.sigmoid(preds)
    mean_prob = probs.mean(dim=1)
    eps = 1e-7
    entropy_map = -mean_prob * torch.log(mean_prob + eps) - (1 - mean_prob) * torch.log(1 - mean_prob + eps)
    return entropy_map.mean().item()

def main():
    grid_conf = {'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5], 'zbound': [-10, 10, 20], 'dbound': [4.0, 45.0, 1.0]}
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
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

    device = torch.device('cuda')
    model = FullDeepEnsembleLSS(grid_conf, data_aug_conf, num_models=5).to(device)
    
    print("Loading weights...")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
    model.eval()

    print("Compiling data...")
    _, loader = compile_data('mini', DATAROOT, data_aug_conf, grid_conf, bsz=1, nworkers=4, parser_name='segmentationdata')

    clean_scores = []
    corrupted_scores = []

    print(f"Collecting scientific stats from {NUM_BATCHES} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= NUM_BATCHES: break
            
            imgs, rots, trans, intrins, post_rots, post_trans, _ = batch
            imgs = imgs.to(device); rots = rots.to(device); trans = trans.to(device)
            intrins = intrins.to(device); post_rots = post_rots.to(device); post_trans = post_trans.to(device)

            # 1. Clean Pass
            preds_clean = model(imgs, rots, trans, intrins, post_rots, post_trans)
            score_clean = calc_entropy_score(preds_clean)
            clean_scores.append(score_clean)

            # 2. Corrupted Pass (Patch Shuffle)
            imgs_corr = apply_patch_shuffle(imgs) # שימוש בפונקציה החדשה
            preds_corr = model(imgs_corr, rots, trans, intrins, post_rots, post_trans)
            score_corr = calc_entropy_score(preds_corr)
            corrupted_scores.append(score_corr)
            
            print(f"Batch {i+1}: Clean={score_clean:.4f}, Shuffled={score_corr:.4f}")

    # === PLOTTING ===
    clean_scores = np.array(clean_scores)
    corrupted_scores = np.array(corrupted_scores)

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(clean_scores, bins=15, alpha=0.7, label='Clean (Low Entropy)', color='blue', density=True)
    plt.hist(corrupted_scores, bins=15, alpha=0.7, label='Shuffled OOD (High Entropy)', color='red', density=True)
    plt.xlabel('Entropy Score')
    plt.title('Distribution Separation')
    plt.legend()
    plt.savefig('scientific_histogram.png')

    # AUROC
    y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(corrupted_scores))])
    y_scores = np.concatenate([clean_scores, corrupted_scores])
    
    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend(loc="lower right")
    plt.title('OOD Detection Performance')
    plt.savefig('scientific_roc.png')

    print("\n" + "="*40)
    print(f"✅ SCIENTIFIC RESULTS:")
    print(f"Avg Clean:     {clean_scores.mean():.4f}")
    print(f"Avg Shuffled:  {corrupted_scores.mean():.4f}")
    print(f"AUROC Score:   {auroc:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()