# Training Scripts

## Quick Start

### Train New Model

```bash
cd /home/shayelbaz/repos/training

python train.py \
    --dataroot ../data/ \
    --version mini \
    --pretrained_path ../checkpoints/model525000.pt \
    --lift_lr_ratio 0.05 \
    --lr 2e-4 \
    --epochs 30 \
    --batch_size 64 \
    --num_models 5 \
    --workers 32 \
    --exp_name my_experiment \
    --seed 42
```

### Key Parameters

- `--dataroot`: Path to NuScenes data (default: `../data/`)
- `--pretrained_path`: Pretrained LSS weights (default: None, use `../checkpoints/model525000.pt`)
- `--lift_lr_ratio`: LR multiplier for Lift components (default: 0.05)
- `--lr`: Base learning rate for Splat (default: 2e-4)
- `--batch_size`: Batch size (default: 12, can go much higher with strong GPUs)
- `--workers`: Data loader workers (default: 8, can go up to 32)
- `--num_models`: Ensemble size (default: 5)

### Output

Checkpoints saved to: `../checkpoints/runs/{version}/{exp_name}/{timestamp}/`

### Monitor Training

```bash
tensorboard --logdir ../checkpoints/runs/
```

