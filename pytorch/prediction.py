import torch
import numpy as np
from torchvision.transforms import functional as TF
from tqdm import tqdm
from skimage import io
import os

def ensemble(X, geometric=True, shuffle=False, n=10):
    """RAMS+ prediction util"""
    if geometric:
        return geometric_ensemble(X, shuffle)
    else:
        return random_ensemble(X, n, shuffle)

def random_ensemble(X, n=10, shuffle=True):
    """RAMS+ prediction util"""
    r = np.zeros((n,2))
    X_ensemble = []
    for i in range(n):
        X_aug, r[i, 0] = flip(X)
        X_aug, r[i, 1] = rotate(X_aug)
        if shuffle:
            X_aug = shuffle_last_axis(X_aug)
        X_ensemble.append(X_aug)
    return torch.stack(X_ensemble), r

def geometric_ensemble(X, shuffle=False):
    """RAMS+ prediction util"""
    r = np.array(np.meshgrid([0, 1], [0, 1, 2, 3])).T.reshape(-1, 2)  # generates all combinations (8) for flip/rotate parameter
    X_ensemble = []
    for i in range(8):
        X_aug = flip(X, r[i, 0])[0]
        X_aug = rotate(X_aug, r[i, 1])[0]
        if shuffle:
            X_aug = shuffle_last_axis(X_aug)
        X_ensemble.append(X_aug)
    return torch.stack(X_ensemble), r

def unensemble(X, r):
    """RAMS+ prediction util"""
    X_unensemble = []
    for i in range(X.size(0)):
        X_aug = rotate(X[i], 4 - r[i, 1])[0]  # to reverse rotation: k2=4-k1
        X_aug = flip(X_aug, r[i, 0])[0]
        X_unensemble.append(X_aug)
    return torch.mean(torch.stack(X_unensemble), dim=0, keepdim=True)

def flip(X, rn=None):
    """flip a tensor"""
    if rn is None:
        rn = torch.rand(1).item()
    return TF.hflip(X) if rn <= 0.5 else X, np.round(rn)

def rotate(X, rn=None):
    """rotate a tensor"""
    if rn is None:
        rn = torch.randint(0, 4, (1,)).item()
    return TF.rotate(X, rn * 90), rn

def shuffle_last_axis(X):
    """shuffle last tensor axis"""
    X = X.permute(3, 2, 1, 0)
    X = X[torch.randperm(X.size(0))]
    X = X.permute(3, 2, 1, 0)
    return X

def predict_tensor(model, x):
    """RAMS prediction util"""
    lr_batch = x.float()
    sr_batch = model(lr_batch)
    sr_batch = torch.clamp(sr_batch, 0, 2**16)
    sr_batch = torch.round(sr_batch).float()
    return sr_batch

def predict_tensor_permute(model, x, n_ens=10):
    """RAMS+ prediction util"""
    lr_batch = x.float()
    sr_batch = []
    for _ in range(n_ens):
        lr = shuffle_last_axis(lr_batch)
        sr_batch.append(model(lr.unsqueeze(0))[0])
    sr_batch = torch.stack(sr_batch)
    sr_batch = torch.clamp(sr_batch, 0, 2**16)
    sr_batch = torch.round(sr_batch).float()
    return torch.mean(sr_batch, dim=0, keepdim=True)

def save_predictions(x, band, submission_dir):
    """RAMS save util"""
    if band == 'NIR':
        i = 1306
    elif band == 'RED':
        i = 1160

    for index in tqdm(range(x.size(0))):
        io.imsave(os.path.join(submission_dir, f'imgset{i}.png'), x[index][0, :, :, 0].numpy().astype(np.uint16),
                  check_contrast=False)
        i += 1

def save_predictions_permute(x, band, submission_dir):
    """RAMS+ save util"""
    if band == 'NIR':
        i = 1306
    elif band == 'RED':
        i = 1160

    for index in tqdm(range(x.size(0))):
        io.imsave(os.path.join(submission_dir, f'imgset{i}.png'), x[index][0,:,:,0].astype(np.uint16),
                  check_contrast=False)
        i+=1
