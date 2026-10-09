"""
Module 2: Dataset Loading, Retinal CLAHE Preprocessing & Balanced Augmentation
"""

import os
import glob
import re
import random
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from .config import cfg

class RetinalDataset(Dataset):
    def __init__(self, paths, labels, is_train=True):
        self.paths = paths
        self.labels = labels
        self.is_train = is_train

        # Authors exact data augmentation parameters (2 deg rotation, zoom, shift, hflip)
        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=3),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fpath = self.paths[idx]
        label = self.labels[idx]

        bgr = cv2.imread(fpath)
        if bgr is None:
            bgr = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        # CLAHE on lightness channel
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        tensor = self.tf(Image.fromarray(rgb))
        return tensor, label

def load_balanced_partitions(dataset_cache_dir):
    all_files = glob.glob(os.path.join(str(dataset_cache_dir), "*", "*", "*.jpg"))
    cat_files = {cat: [] for cat in cfg.CATEGORIES}

    for fpath in all_files:
        m = re.search(r'DG(\d+)', fpath)
        if m:
            dg = f"DG{m.group(1)}"
            if dg in cat_files:
                cat_files[dg].append(fpath)

    TARGET_PER_CLASS = 800
    expanded_paths = []
    expanded_labels = []
    cat_to_idx = {c: i for i, c in enumerate(cfg.CATEGORIES)}

    for cat in cfg.CATEGORIES:
        files = cat_files[cat]
        if not files:
            continue
        idx = cat_to_idx[cat]
        expanded_paths.extend(files)
        expanded_labels.extend([idx] * len(files))

        needed = max(0, TARGET_PER_CLASS - len(files))
        if needed > 0:
            oversampled = random.choices(files, k=needed)
            expanded_paths.extend(oversampled)
            expanded_labels.extend([idx] * len(oversampled))

    # Split: 80% Train, 10% Val, 10% Test
    X_tv, X_test, y_tv, y_test = train_test_split(
        expanded_paths, expanded_labels, test_size=0.10, stratify=expanded_labels, random_state=cfg.SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.111, stratify=y_tv, random_state=cfg.SEED
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), cat_to_idx
