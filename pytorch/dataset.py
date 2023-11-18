import torch
from torch.utils.data import Dataset, DataLoader

import random
import torchvision.transforms.functional as TF

class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x, y, y_mask):
        # Apply transform to x
        x, y, y_mask = self.transform(x, y, y_mask)
        
        return x, y, y_mask


class RandomRotateFlipTransform:
    def __init__(self, rotation_chance=0.5, flip_chance=0.5):
        self.rotation_chance = rotation_chance
        self.flip_chance = flip_chance

    def __call__(self, x, y, y_mask):
        # Random rotation
        if random.random() < self.rotation_chance:
            k = random.choice([0, 1, 2, 3])  # 0: 0°, 1: 90°, 2: 180°, 3: 270°
            x = torch.rot90(x, k, [1, 2])
            y = torch.rot90(y, k, [1, 2])
            y_mask = torch.rot90(y_mask, k, [1, 2])
        
        # Random horizontal and/or vertical flip
        if random.random() < self.flip_chance:
            # Apply horizontal flip
            if random.random() < 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])
                y_mask = torch.flip(y_mask, [2])
            
            # Apply vertical flip
            if random.random() < 0.5:
                x = torch.flip(x, [1])
                y = torch.flip(y, [1])
                y_mask = torch.flip(y_mask, [1])
        
        return x, y, y_mask


class SuperResolutionDataset(Dataset):
    def __init__(self, X_data, y_data, y_mask, transform = None):
        self.X_data = X_data
        self.y_data = y_data
        self.y_mask = y_mask
        self.transform = transform

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        X = self.X_data[idx]
        y = self.y_data[idx]
        y_mask = self.y_mask[idx]

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.from_numpy(X).permute(2, 0, 1).float()  # [C, H, W]
        y_tensor = torch.from_numpy(y).permute(2, 0, 1).float()  # [C, H, W]
        y_mask_tensor = torch.from_numpy(y_mask).permute(2, 0, 1).float()  # [C, H, W]

        if self.transform:
            X_tensor, y_tensor, y_mask_tensor = self.transform(X_tensor, y_tensor, y_mask_tensor)

        return X_tensor, y_tensor, y_mask_tensor
