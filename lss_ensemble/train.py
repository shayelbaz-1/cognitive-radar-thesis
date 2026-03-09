import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
import logging
import shutil
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler 
import torchvision.transforms.functional as TF 

import PIL.Image
import numpy as np

# === FIX FOR PILLOW 10.0.0+ ===
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Adjust path if needed (now running from training/ folder)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lift-splat-shoot'))
from src.models import LiftSplatShoot
from src.data import compile_data

# ==========================================
#          CONFIGURATION & ARGS
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description="Hybrid Deep Ensemble with Visual Checks")
    
    parser.add_argument('--dataroot', type=str, default='./data/', help='Path to NuScenes root')
    parser.add_argument('--version', type=str, default='mini', choices=['mini', 'trainval'])
    parser.add_argument('--exp_name', type=str, default='hybrid_vis_run', help='Name of the experiment')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=12) 
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_models', type=int, default=5, help='Number of ensemble heads (models)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained LSS weights')
    parser.add_argument('--train_from_scratch', action='store_true', 
                        help='Train from scratch, ignoring any pretrained weights')
    parser.add_argument('--lift_lr_ratio', type=float, default=0.05, help='LR ratio for Lift components (vs Splat)')
    
    return parser.parse_args()

def setup_logger(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ==========================================
#        HYBRID DEEP ENSEMBLE MODEL
# ==========================================
class FullDeepEnsembleLSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5, pretrained_path=None, train_from_scratch=False):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList()
        
        print(f"🧠 Building Hybrid Ensemble ({num_models} heads/models)...")
        
        # Load pretrained weights once if provided and not training from scratch
        pretrained_dict = None
        if train_from_scratch:
            print(f"✨ Training from scratch - initializing all weights randomly")
        elif pretrained_path and os.path.exists(pretrained_path):
            print(f"📦 Loading pretrained weights from {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        else:
            print(f"⚠️  No pretrained weights provided - training from scratch")
        
        for i in range(num_models):
            m = LiftSplatShoot(grid_conf, data_aug_conf, outC=outC)
            
            # Initialize with pretrained Lift-Splat weights if available and not training from scratch
            if pretrained_dict is not None and not train_from_scratch:
                m.load_state_dict(pretrained_dict, strict=False)
                print(f"   Model {i+1}/{num_models}: Loaded pretrained weights")
            else:
                print(f"   Model {i+1}/{num_models}: Random initialization")
            
            # Reset BEV encoder for ensemble diversity (always done for diversity among heads)
            self._reset_weights(m.bevencode) 
            self.models.append(m)
            
    def _reset_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        outputs = []
        for model in self.models:
            res = model(x, rots, trans, intrins, post_rots, post_trans)
            outputs.append(res)
        return torch.stack(outputs, dim=1)

# ==========================================
#             TRAINER CLASS
# ==========================================

class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Save to parent directory's checkpoints folder
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.exp_dir = os.path.join(parent_dir, 'checkpoints', 'runs', args.version, args.exp_name, current_time)
        self.ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = setup_logger(self.exp_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]).to(self.device))
        
        # Differential Learning Rates: Slower for Lift (pretrained), faster for Splat
        # When training from scratch, use same LR for all components
        raw_model = model.module if hasattr(model, "module") else model
        
        if args.train_from_scratch or args.pretrained_path is None:
            # Training from scratch: use same learning rate for all parameters
            all_params = []
            for ensemble_model in raw_model.models:
                all_params.extend(ensemble_model.parameters())
            
            self.optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
            self.logger.info(f"📊 Uniform LR (training from scratch): {args.lr:.2e}")
        else:
            # Using pretrained weights: use differential learning rates
            lift_params = []
            splat_params = []
            
            for ensemble_model in raw_model.models:
                # Lift components (camera encoder + depth)
                lift_params.extend(ensemble_model.camencode.parameters())
                # Splat component (BEV encoder)
                splat_params.extend(ensemble_model.bevencode.parameters())
            
            self.optimizer = optim.Adam([
                {'params': lift_params, 'lr': args.lr * args.lift_lr_ratio, 'name': 'lift'},
                {'params': splat_params, 'lr': args.lr, 'name': 'splat'}
            ], weight_decay=args.weight_decay)
            
            self.logger.info(f"📊 Differential LR: Lift={args.lr * args.lift_lr_ratio:.2e}, Splat={args.lr:.2e}")

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        self.scaler = GradScaler()
        self.logger.info(f"🚀 Training Started! Checkpoints: {self.ckpt_dir}")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", unit="batch")
        for i, batch in enumerate(pbar):
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            
            imgs = imgs.to(self.device)
            binimgs = binimgs.to(self.device)
            rots, trans, intrins = rots.to(self.device), trans.to(self.device), intrins.to(self.device)
            post_rots, post_trans = post_rots.to(self.device), post_trans.to(self.device)

            self.optimizer.zero_grad()
            
            with autocast():
                preds_stack = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
                loss = 0
                for m in range(self.args.num_models):
                    loss += self.criterion(preds_stack[:, m], binimgs)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            avg_loss = loss.item() / self.args.num_models
            running_loss += avg_loss
            pbar.set_postfix(loss=avg_loss)
            self.writer.add_scalar('Loss/Train', avg_loss, (epoch - 1) * len(self.train_loader) + i)

        return running_loss / len(self.train_loader)

    def validate_and_visualize(self, epoch):
        self.model.eval()
        running_loss = 0.0
        
        vis_batch = None # To store data for visualization
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
                
                imgs = imgs.to(self.device); binimgs = binimgs.to(self.device)
                rots, trans, intrins = rots.to(self.device), trans.to(self.device), intrins.to(self.device)
                post_rots, post_trans = post_rots.to(self.device), post_trans.to(self.device)

                with autocast():
                    preds_stack = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
                    loss = 0
                    for m in range(self.args.num_models):
                        loss += self.criterion(preds_stack[:, m], binimgs)
                
                running_loss += (loss.item() / self.args.num_models)
                
                # שמור את הבאץ' הראשון לויזואליזציה (כדי שיהיה עקבי)
                if i == 0:
                    vis_batch = (imgs, preds_stack, binimgs)

        # --- LOGGING VISUALS & ENTROPY STATS ---
        if vis_batch is not None:
            self.log_visuals(epoch, *vis_batch)
            self.log_entropy_stats(epoch, vis_batch[1], vis_batch[2])

        avg_loss = running_loss / len(self.val_loader)
        self.writer.add_scalar('Performance/Val_Loss', avg_loss, epoch)
        
        return avg_loss

    def log_visuals(self, epoch, imgs, preds_stack, gt):
        """
        Create a dashboard: [Input] [Prediction] [Ground Truth] [Entropy]
        """
        # 1. Input Image (CAM_FRONT, index 1)
        img_vis = imgs[0, 1].clone()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        
        # 2. Prediction (Ensemble Mean)
        probs = torch.sigmoid(preds_stack)
        mean_prob = probs.mean(dim=1) # (B, 1, H, W)
        pred_vis = mean_prob[0] # (1, H, W)
        
        # 3. Ground Truth
        gt_vis = gt[0] # (1, H, W)
        
        # 4. Entropy
        eps = 1e-7
        entropy = -mean_prob * torch.log(mean_prob + eps) - (1 - mean_prob) * torch.log(1 - mean_prob + eps)
        ent_vis = entropy[0]
        ent_vis = (ent_vis - ent_vis.min()) / (ent_vis.max() - ent_vis.min() + 1e-5)

        # === התיקון: המרת ערוצים והתאמת גדלים (RESIZE) ===
        
        # המרה ל-3 ערוצים (RGB)
        pred_vis = pred_vis.repeat(3, 1, 1)
        gt_vis = gt_vis.repeat(3, 1, 1)
        ent_vis = ent_vis.repeat(3, 1, 1)

        # בדיקת הגובה של מפת ה-BEV (שאמור להיות 200)
        bev_h, bev_w = pred_vis.shape[1], pred_vis.shape[2]
        
        # שינוי גודל תמונת המצלמה (שהיא 128) לגובה של ה-BEV (שהוא 200)
        # שימוש ב-interpolate דורש מימד נוסף (Batch), לכן unsqueeze ו-squeeze
        import torch.nn.functional as F
        img_vis = F.interpolate(img_vis.unsqueeze(0), size=(bev_h, 352), mode='bilinear', align_corners=False).squeeze(0)

        # עכשיו כולם באותו גובה, אפשר לחבר
        grid = torch.cat([img_vis, pred_vis, gt_vis, ent_vis], dim=2) 
        
        self.writer.add_image('Dashboard/Input_Pred_GT_Entropy', grid, epoch)
    
    def log_entropy_stats(self, epoch, preds_stack, gt):
        """
        Log detailed entropy statistics for analysis
        """
        probs = torch.sigmoid(preds_stack)
        mean_prob = probs.mean(dim=1)
        
        # Compute entropy
        eps = 1e-7
        entropy = -mean_prob * torch.log(mean_prob + eps) - (1 - mean_prob) * torch.log(1 - mean_prob + eps)
        entropy = entropy.flatten().float()  # Ensure float dtype for quantile
        
        # Statistics
        ent_mean = entropy.mean().item()
        ent_std = entropy.std().item()
        ent_min = entropy.min().item()
        ent_max = entropy.max().item()
        ent_percentiles = torch.quantile(entropy, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32).to(entropy.device))
        
        # Log to tensorboard
        self.writer.add_histogram('Entropy/Distribution', entropy, epoch)
        self.writer.add_scalar('Entropy/Mean', ent_mean, epoch)
        self.writer.add_scalar('Entropy/Std', ent_std, epoch)
        self.writer.add_scalar('Entropy/Min', ent_min, epoch)
        self.writer.add_scalar('Entropy/Max', ent_max, epoch)
        self.writer.add_scalar('Entropy/Q25', ent_percentiles[0].item(), epoch)
        self.writer.add_scalar('Entropy/Median', ent_percentiles[1].item(), epoch)
        self.writer.add_scalar('Entropy/Q75', ent_percentiles[2].item(), epoch)
        
        # High entropy ratio (potential radar targets)
        high_entropy_threshold = ent_mean + ent_std
        high_entropy_ratio = (entropy > high_entropy_threshold).float().mean().item()
        self.writer.add_scalar('Entropy/HighEntropyRatio', high_entropy_ratio, epoch)
        
        # Ensemble disagreement (variance across models)
        variance = probs.var(dim=1).flatten()
        var_mean = variance.mean().item()
        self.writer.add_scalar('Ensemble/Variance_Mean', var_mean, epoch)
        
        self.logger.info(f"Entropy Stats | Mean: {ent_mean:.4f} | Std: {ent_std:.4f} | High%: {high_entropy_ratio*100:.1f}%")

    def run(self):
        best_loss = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            self.logger.info(f"--- Epoch {epoch}/{self.args.epochs} ---")
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_and_visualize(epoch)
            
            self.logger.info(f"Results | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, True)
            elif epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch, val_loss, False)
                
        self.writer.close()

    def save_checkpoint(self, epoch, loss, is_best):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            'epoch': epoch,
            'state_dict': raw_model.state_dict(),
            'loss': loss,
        }
        path = os.path.join(self.ckpt_dir, f"checkpoint_ep{epoch}.pth")
        torch.save(state, path)
        if is_best: shutil.copyfile(path, os.path.join(self.ckpt_dir, "model_best.pth"))

# ==========================================
#                  MAIN
# ==========================================
def main():
    args = get_args()
    set_seed(args.seed)
    
    print("=" * 70)
    print(f"🚀 TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Dataset Version: {args.version.upper()}")
    print(f"  Experiment Name: {args.exp_name}")
    print(f"  Number of Ensemble Heads: {args.num_models}")
    print(f"  Training Mode: {'FROM SCRATCH ✨' if args.train_from_scratch else 'With Pretrained Weights 📦' if args.pretrained_path else 'FROM SCRATCH (no weights provided)'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Lift LR Ratio: {args.lift_lr_ratio}")
    print("=" * 70)
    print()
    
    print(f"🔄 Preparing Data [{args.version.upper()}]...")
    
    grid_conf = {
        'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5],
        'zbound': [-10, 10, 20], 'dbound': [4.0, 45.0, 1.0],
    }

    # Clean config for everything (Model learns CLEAN world)
    aug_conf = {
        'resize_lim': (0.193, 0.225), 'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600,
        'bot_pct_lim': (0.0, 0.0), 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6, 
        'rand_flip': True 
    }
    
    val_aug_conf = aug_conf.copy()
    val_aug_conf['rand_flip'] = False 

    print("   -> Compiling Loaders...")
    train_loader, _ = compile_data(args.version, args.dataroot, aug_conf, grid_conf, bsz=args.batch_size, nworkers=args.workers, parser_name='segmentationdata')
    _, val_loader = compile_data(args.version, args.dataroot, val_aug_conf, grid_conf, bsz=args.batch_size, nworkers=args.workers, parser_name='segmentationdata')
    
    model = FullDeepEnsembleLSS(
        grid_conf, aug_conf, outC=1, 
        num_models=args.num_models,
        pretrained_path=args.pretrained_path,
        train_from_scratch=args.train_from_scratch
    )
    trainer = Trainer(args, model, train_loader, val_loader)
    trainer.run()

if __name__ == "__main__":
    main()