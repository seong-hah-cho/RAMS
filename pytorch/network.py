import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Settings from the paper
MEAN = 7433.6436  # mean of the Proba-V dataset
STD = 2353.0723  # std of the Proba-V dataset

# Normalization functions
def normalize(x):
    """Normalize tensor"""
    # Paper: The input images are normalized using the mean and standard deviation of the dataset.
    return (x - MEAN) / STD


def denormalize(x):
    """Denormalize tensor"""
    # Paper: The output images are denormalized before evaluation.
    return x * STD + MEAN


# Reflective padding
class ReflectivePadding3d(nn.Module):
    """Reflecting padding on H and W dimension"""
    # Paper: Reflective padding is used to avoid border artifacts in the convolution operations.
    def __init__(self, padding):
        super().__init__()
        self.padding = padding  # Padding on the last 3 dimensions (D, H, W)

    def forward(self, x):
        # Only pad the last three dimensions of the input tensor (D, H, W)
        return F.pad(x, self.padding, mode='reflect')


# Convolution layers with weight normalization
def conv3d_weightnorm(in_channels, out_channels, kernel_size, padding='same', activation=None):
    """
    3D convolution with weight normalization.
    According to the paper, weight normalization is applied to stabilize the training
    by reparameterizing the weight vectors in a neural network.
    """
    if padding == 'same':
        padding = kernel_size // 2
    conv3d = weight_norm(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding))
    return conv3d


def conv2d_weightnorm(in_channels, out_channels, kernel_size, padding='same', activation=None):
    """
    2D convolution with weight normalization.
    Weight normalization helps in faster convergence and stabilizes the training process.
    """
    if padding == 'same':
        padding = kernel_size // 2
    conv2d = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
    return conv2d


# Residual Feature Attention Block (RFAB)
class RFAB(nn.Module):
    """
    Residual Feature Attention Block (RFAB).
    The paper describes this block as a way to focus on informative features
    and suppress less useful ones by using attention mechanisms.
    """
    def __init__(self, in_channels, out_channels, kernel_size, r, activation='relu'):
        super(RFAB, self).__init__()
        self.conv3d_1 = conv3d_weightnorm(in_channels, out_channels, kernel_size, padding='same')
        self.conv3d_2 = conv3d_weightnorm(out_channels, out_channels, kernel_size, padding='same')
        self.conv3d_downscale = conv3d_weightnorm(out_channels, out_channels // r, (1, 1, 1), padding=0)
        self.conv3d_upscale = conv3d_weightnorm(out_channels // r, out_channels, (1, 1, 1), padding=0)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

    def forward(self, x):
        x_res = x
        x = self.conv3d_1(x)
        x = self.activation(x)
        x = self.conv3d_2(x)

        # Adaptive pooling to a single value per channel
        x_to_scale = x
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = self.conv3d_downscale(x)
        x = self.activation(x)
        x = self.conv3d_upscale(x)
        x = torch.sigmoid(x)

        # Element-wise multiply (scaling) and residual connection
        x_scaled = x_to_scale * x
        return x_scaled + x_res

# Residual Temporal Attention Block (RTAB)
class RTAB(nn.Module):
    """
    Residual Temporal Attention Block (RTAB).
    Similar to RFAB, but operates on 2D features, applying attention across temporal features
    to enhance feature representation for improved super-resolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, r, activation='relu'):
        super(RTAB, self).__init__()
        self.conv2d_1 = conv2d_weightnorm(in_channels, out_channels, kernel_size, padding='same')
        self.conv2d_2 = conv2d_weightnorm(out_channels, out_channels, kernel_size, padding='same')
        self.conv2d_downscale = conv2d_weightnorm(out_channels, out_channels // r, (1, 1), padding=0)
        self.conv2d_upscale = conv2d_weightnorm(out_channels // r, out_channels, (1, 1), padding=0)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

    def forward(self, x):
        x_res = x
        x = self.conv2d_1(x)
        x = self.activation(x)
        x = self.conv2d_2(x)

        # Adaptive pooling to a single value per feature map
        x_to_scale = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv2d_downscale(x)
        x = self.activation(x)
        x = self.conv2d_upscale(x)
        x = torch.sigmoid(x)

        # Element-wise multiply (scaling) and residual connection
        x_scaled = x_to_scale * x

        return x_scaled + x_res

class RAMS(nn.Module):
    """
    RAMS Deep Neural Network for Multi-Image Super Resolution of Remotely Sensed Images.
    The network architecture combines residual learning, attention mechanisms, and
    a global residual path to facilitate the learning process and improve the quality
    of super-resolved images.
    """
    def __init__(self, scale, in_channels, out_channels, kernel_size, r, N):
        super(RAMS, self).__init__()
        self.scale = scale
        self.N = N

        # Initial padding
        self.initial_padding = nn.ReflectionPad2d((1, 1, 1, 1, 0, 0,))

        # Low-level features extraction
        self.conv_in = conv3d_weightnorm(1, out_channels, kernel_size, padding='same')

        # Residual blocks
        self.RFAB_blocks = nn.ModuleList([RFAB(out_channels, out_channels, kernel_size, r) for _ in range(N)])
        self.conv_out = conv3d_weightnorm(out_channels, out_channels, kernel_size, padding='same')

        # Temporal Reduction
        self.temporal_reduction_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad3d((1, 1, 1, 1, 0, 0)),
                RFAB(out_channels, out_channels, kernel_size, r),
                conv3d_weightnorm(out_channels, out_channels, kernel_size, padding='valid', activation='relu')
            )
            for _ in range(int((in_channels - 1) / (kernel_size - 1) - 1))
        ])

        # Upscaling
        self.upscale_conv = conv3d_weightnorm(out_channels, scale ** 2, (3, 3, 3), padding='valid')
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Global residual path
        self.global_residual = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            RTAB(in_channels, in_channels, kernel_size, r),
            conv2d_weightnorm(in_channels, scale ** 2, (3, 3), padding='valid'),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # Normalize input
        x = normalize(x)  # Assuming MEAN and STD are defined as constants

        # Global residual path
        x_global_res = self.global_residual(x)

        # Expand and initial padding
        x = x.unsqueeze(1)  # PyTorch uses (B, C, D, H, W) for 3D convolutions
        x = self.initial_padding(x)

        # Extract features
        x = self.conv_in(x)
        x_res = x

        # Apply RFAB blocks
        for i in range(self.N):
            x = self.RFAB_blocks[i](x)

        # Apply the final convolution
        x = self.conv_out(x)
        x = x + x_res

        # Temporal reduction
        for block in self.temporal_reduction_blocks:
            x = block(x)

        # Upscaling
        x = self.upscale_conv(x)
        x = x.squeeze(2)  # Remove the depth dimension which is now 1
        x = self.pixel_shuffle(x)

        # Global and local path combination
        x = x + x_global_res

        # Denormalize output
        x = denormalize(x)

        return x
