import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_pipeline, preprocess_mask


# Augmentations entraînement
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
])


class SkinDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        assert len(image_paths) == len(mask_paths), \
            f"Nombre d'images ({len(image_paths)}) != masques ({len(mask_paths)})"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess_pipeline(self.image_paths[idx])   # (128,128,3) float32
        mask  = preprocess_mask(self.mask_paths[idx])        # (128,128)   float32

        if self.transform is not None:
            augmented = self.transform(image=image.astype(np.float32), mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # (H,W,C) → (C,H,W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # (H,W) → (1,H,W)
        mask_tensor  = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor