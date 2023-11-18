import os
import random

import numpy as np

import torch

from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import functional as TF


def random_flip(lr_img, hr_img, hr_img_mask):
    """Data Augmentation: flip data samples randomly"""
    if random.random() > 0.5:
        lr_img = TF.hflip(lr_img)
        hr_img = TF.hflip(hr_img)
        hr_img_mask = TF.hflip(hr_img_mask)
    return lr_img, hr_img, hr_img_mask


def random_rotate(lr_img, hr_img, hr_img_mask):
    """Data Augmentation: rotate data samples randomly of a 90 degree angle"""
    angle = random.choice([0, 90, 180, 270])
    lr_img = TF.rotate(lr_img, angle)
    hr_img = TF.rotate(hr_img, angle)
    hr_img_mask = TF.rotate(hr_img_mask, angle)
    return lr_img, hr_img, hr_img_mask


class Trainer:
    def __init__(self, model, loss_fn, metric_fn, optimizer, device, 
                 checkpoint_dir='./checkpoint', batch_size=4, shuffle=True, num_workers=0):

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir

        # Create directories if they do not exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize best PSNR value for checkpointing
        self.best_psnr = 0.0

    def train_step(self, lr, hr, mask):
        self.model.train()
        lr, hr, mask = lr.to(self.device), hr.to(self.device), mask.to(self.device)

        sr = self.model(lr)
        loss = self.loss_fn(sr, hr, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metric = self.metric_fn(sr, hr, mask)
        return loss.item(), metric

    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_metric = 0.0
            for lr, hr, mask in val_loader:
                lr, hr, mask = lr.to(self.device), hr.to(self.device), mask.to(self.device)
                sr = self.model(lr)
                loss = self.loss_fn(sr, hr, mask)
                metric = self.metric_fn(sr, hr, mask)
                val_loss += loss.item()
                val_metric += metric
            val_loss /= len(val_loader)
            val_metric /= len(val_loader)
        return val_loss, val_metric

    def fit(self, train_dataset, val_dataset, epochs, evaluate_every=5):
        # Data loaders for training and validation datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        global_step = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_metric = 0.0
            for lr, hr, mask in train_loader:
                loss, metric = self.train_step(lr, hr, mask)
                epoch_loss += loss
                epoch_metric += metric
                global_step += 1

            # Average loss and metric for the epoch
            epoch_loss /= len(train_loader)
            epoch_metric /= len(train_loader)

            print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f} Metric: {epoch_metric:.4f}")

            # Validation
            if epoch % evaluate_every == 0:
                val_loss, val_metric = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f} Metric: {val_metric:.4f}")

                # Save checkpoint if there is improvement
                if val_metric > self.best_psnr:
                    self.best_psnr = val_metric
                    torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/best_model.pth")
                    print("Saved new best checkpoint.")


import pytorch_lightning as pl
from torch.utils.data import DataLoader


class RAMSLightning(pl.LightningModule):
    def __init__(self, model, loss_fn, metric_fn, train_dataset, val_dataset, optimizer_class, optimizer_params):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training step
        lr, hr, mask = batch
        sr = self(lr)
        loss = self.loss_fn(sr, hr, mask)
        metric = self.metric_fn(sr, hr, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        lr, hr, mask = batch
        sr = self(lr)
        loss = self.loss_fn(sr, hr, mask)
        metric = self.metric_fn(sr, hr, mask)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_metric', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict(self, X, index):
        # Convert to PyTorch tensor and add batch dimension if needed
        X_batch = X[index:index+1].astype(np.float32)  # Assuming index is to select a single example
        if len(X_batch.shape) == 3:
            X_batch = X_batch[np.newaxis, ...]  # Add batch dimension
        input_tensor = torch.from_numpy(X_batch).float()
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # PyTorch uses NCHW format

        # Ensure model is in evaluation mode
        self.eval()

        # Predict with PyTorch Lightning model
        with torch.no_grad():
            # Move the input tensor to the same device as the model
            input_tensor = input_tensor.to(self.device)
            # Forward pass
            prediction = self(input_tensor)
    
        # Convert the prediction to numpy array for visualization
        return prediction.cpu().numpy()

    def configure_optimizers(self):
        # Configure optimizers
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        return optimizer

    def train_dataloader(self):
        # Return train dataloader
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers = 4)

    def val_dataloader(self):
        # Return validation dataloader
        return DataLoader(self.val_dataset, batch_size=128, num_workers = 4)