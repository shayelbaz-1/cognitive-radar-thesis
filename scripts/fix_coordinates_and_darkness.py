"""
Quick script to check NuScenes BEV coordinate system and find truly dark scenes
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'lift-splat-shoot'))
from src.data import compile_data

# Check a few scenes
dataroot = './data/'
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

# Load TRAIN data to look for night scenes
train_loader, _ = compile_data('mini', dataroot, data_aug_conf, grid_conf, bsz=1, nworkers=0, parser_name='segmentationdata')

print("Analyzing scenes for brightness...")
brightness_list = []
scene_indices = []

for i, batch in enumerate(train_loader):
    if i >= 100:
        break
    imgs = batch[0]
    brightness = imgs.mean().item()
    brightness_list.append(brightness)
    scene_indices.append(i)
    
    if i % 20 == 0:
        print(f"Scene {i}: brightness = {brightness:.4f}")

# Find truly dark scenes
brightness_array = np.array(brightness_list)
sorted_idx = np.argsort(brightness_array)

print("\n" + "="*60)
print("BRIGHTNESS ANALYSIS")
print("="*60)
print(f"Darkest scene: {brightness_array[sorted_idx[0]]:.4f}")
print(f"Median: {np.median(brightness_array):.4f}")
print(f"Brightest scene: {brightness_array[sorted_idx[-1]]:.4f}")
print(f"\nDarkest 10 scenes:")
for i in range(10):
    idx = sorted_idx[i]
    print(f"  Scene {idx}: {brightness_array[idx]:.4f}")

# Check if there are any truly dark scenes (night)
night_threshold = 0.15  # Adjust based on what you see
night_scenes = brightness_array < night_threshold
print(f"\nScenes with brightness < {night_threshold}: {night_scenes.sum()}")

if night_scenes.sum() > 0:
    print("Found night scenes!")
else:
    print("No truly dark scenes in mini dataset - all are daytime")
    print("Recommendation: Use relative darkness (darkest available)")

