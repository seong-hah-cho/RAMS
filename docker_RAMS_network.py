# import utils and basic libraries
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.preprocessing import gen_sub, bicubic
from utils.loss import l1_loss, psnr, ssim
from utils.network import RAMS, RAMS_RTAB
from utils.training import Trainer
from skimage import io
from zipfile import ZipFile

# GPU settings (we strongly discouraged to run this notebook without an available GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#-------------
# General Settings
#-------------
PATH_DATASET = 'dataset' # pre-processed dataset path
name_net = 'RAMS' # name of the network
LR_SIZE = 32 # pathces dimension
SCALE = 3 # upscale of the proba-v dataset is 3
HR_SIZE = LR_SIZE * SCALE # upscale of the dataset is 3
OVERLAP = 32 # overlap between pathces
CLEAN_PATH_PX = 0.85 # percentage of clean pixels to accept a patch
band = 'NIR' # choose the band for the training
checkpoint_dir = f'ckpt/{band}_{name_net}_retrain' # weights path
log_dir = 'logs' # tensorboard logs path
submission_dir = 'submission' # submission dir


#-------------
# Network Settings
#-------------
FILTERS = 32 # features map in the network
KERNEL_SIZE = 3 # convolutional kernel size dimension (either 3D and 2D)
CHANNELS = 9 # number of temporal steps
R = 8 # attention compression
N = 12 # number of residual feature attention blocks
lr = 1e-4 # learning rate (Nadam optimizer)
BATCH_SIZE = 32 # batch size
EPOCHS_N = 100 # number of epochs


# Create logs folder
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


# Load training dataset
X_train = np.load(os.path.join(PATH_DATASET, f'X_{band}_train.npy'))
y_train = np.load(os.path.join(PATH_DATASET, f'y_{band}_train.npy'))
y_train_mask = np.load(os.path.join(PATH_DATASET, f'y_{band}_train_masks.npy'))


# Load validation dataset
X_val = np.load(os.path.join(PATH_DATASET, f'X_{band}_val.npy'))
y_val = np.load(os.path.join(PATH_DATASET, f'y_{band}_val.npy'))
y_val_mask = np.load(os.path.join(PATH_DATASET, f'y_{band}_val_masks.npy'))


# Create patches for LR images
d = LR_SIZE  # 32x32 patches
s = OVERLAP  # overlapping patches
# Ex: n = (128-d)/s+1 = 7 -> 49 sub images from each image

X_train_patches = gen_sub(X_train,d,s)
X_val_patches = gen_sub(X_val,d,s)


# Create patches for HR images and masks
d = HR_SIZE  # 96x96 patches
s = OVERLAP * SCALE  # overlapping patches
# Ex: n = (384-d)/s+1 = 7 -> 49 sub images from each image

y_train_patches = gen_sub(y_train,d,s)
y_train_mask_patches = gen_sub(y_train_mask,d,s)

y_val_patches = gen_sub(y_val,d,s)
y_val_mask_patches = gen_sub(y_val_mask,d,s)


# Free up memory
del X_train, y_train, y_train_mask

del X_val, y_val, y_val_mask


# Find patches indices with a lower percentage of clean pixels in train array
patches_to_remove_train = [i for i,m in enumerate(y_train_mask_patches) if np.count_nonzero(m)/(HR_SIZE*HR_SIZE) < CLEAN_PATH_PX]


# Find patches indices with a lower percentage of clean pixels in validation array
patches_to_remove_val = [i for i,m in enumerate(y_val_mask_patches) if np.count_nonzero(m)/(HR_SIZE*HR_SIZE) < CLEAN_PATH_PX]


# Remove patches not clean
X_train_patches = np.delete(X_train_patches,patches_to_remove_train,axis=0)
y_train_patches =  np.delete(y_train_patches,patches_to_remove_train,axis=0)
y_train_mask_patches =  np.delete(y_train_mask_patches,patches_to_remove_train,axis=0)

X_val_patches = np.delete(X_val_patches,patches_to_remove_val,axis=0)
y_val_patches =  np.delete(y_val_patches,patches_to_remove_val,axis=0)
y_val_mask_patches =  np.delete(y_val_mask_patches,patches_to_remove_val,axis=0)


# Build rams network
rams_network = RAMS(scale=SCALE, filters=FILTERS, 
                 kernel_size=KERNEL_SIZE, channels=CHANNELS, r=R, N=N)


trainer_rams = Trainer(rams_network, band, HR_SIZE, name_net,
                      loss=l1_loss,
                      metric=psnr,
                      optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
                      checkpoint_dir=os.path.join(checkpoint_dir),
                      log_dir=log_dir)


trainer_rams.fit(X_train_patches,
                [y_train_patches.astype('float32'), y_train_mask_patches], initial_epoch = 0,
                batch_size=BATCH_SIZE, evaluate_every=400, data_aug = True, epochs=EPOCHS_N,
                validation_data=(X_val_patches, [y_val_patches.astype('float32'), y_val_mask_patches])) 