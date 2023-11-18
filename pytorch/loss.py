import torch

import torch.nn.functional as F
from torch.nn.functional import pad

HR_SIZE = 96

def log10(x):
    """
    Compute log base 10 using PyTorch
    """
    numerator = torch.log(x)
    denominator = torch.log(torch.tensor(10.0, dtype=numerator.dtype, device=numerator.device))
    return numerator / denominator


def l1_loss(y_true, y_pred, y_mask, HR_SIZE=96):
    """
    Modified L1 loss to take into account pixel shifts, implemented in PyTorch.
    """
    border = 3
    max_pixel_shifts = 2 * border
    size_cropped_image = HR_SIZE - max_pixel_shifts
    cropped_predictions = y_pred[:, :, border:HR_SIZE - border, border:HR_SIZE - border]

    X = []
    for i in range(max_pixel_shifts + 1):  # range(7)
        for j in range(max_pixel_shifts + 1):  # range(7)
            cropped_labels = y_true[:, :, i:i + size_cropped_image, j:j + size_cropped_image]
            cropped_masks = y_mask[:, :, i:i + size_cropped_image, j:j + size_cropped_image]

            # Apply mask
            cropped_predictions_masked = cropped_predictions * cropped_masks
            cropped_labels_masked = cropped_labels * cropped_masks

            total_pixels_masked = cropped_masks.sum(dim=[2, 3])

            # Bias brightness
            b = (1.0 / total_pixels_masked) * (cropped_labels_masked - cropped_predictions_masked).sum(dim=[2, 3])

            # Reshape b to be broadcastable
            b = b.view(-1, 1, 1, 1)

            # Apply bias correction to predictions
            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions *= cropped_masks

            # Calculate L1 loss
            l1_loss_value = (1.0 / total_pixels_masked) * torch.abs(cropped_labels_masked - corrected_cropped_predictions).sum(dim=[2, 3])
            X.append(l1_loss_value)

    X = torch.stack(X)
    min_l1 = torch.min(X, dim=0)[0]

    return min_l1.mean()

def psnr(y_true, y_pred, y_mask, size_image=HR_SIZE):
    """
    Modified PSNR metric to take into account pixel shifts, using PyTorch
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_cropped_image = size_image - max_pixels_shifts
    cropped_predictions = y_pred[:, :, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, :, i:i + size_cropped_image, j:j + size_cropped_image]
            cropped_masks = y_mask[:, :, i:i + size_cropped_image, j:j + size_cropped_image]

            cropped_masks = cropped_masks.float()

            cropped_predictions_masked = cropped_predictions.float() * cropped_masks
            cropped_labels_masked = cropped_labels.float() * cropped_masks

            total_pixels_masked = cropped_masks.sum(dim=[2, 3])

            # bias brightness
            b = (1.0 / total_pixels_masked) * (cropped_labels_masked - cropped_predictions_masked).sum(dim=[2, 3])

            b = b.view(-1, 1, 1, 1)

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions *= cropped_masks

            corrected_mse = (1.0 / total_pixels_masked) * torch.pow(cropped_labels_masked - corrected_cropped_predictions, 2).sum(dim=[2, 3])

            # Peak Signal-to-Noise Ratio calculation
            cPSNR = 10.0 * log10((65535.0 ** 2) / corrected_mse)
            X.append(cPSNR)

    X = torch.stack(X)
    max_cPSNR = torch.max(X, dim=0)[0]
    return torch.mean(max_cPSNR)


def structural_similarity(y_true, y_pred, data_range):
    """
    Compute the SSIM (Structural Similarity Index) between two images.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(y_true, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y_pred, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(y_true ** 2, kernel_size=3, stride=1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y_pred ** 2, kernel_size=3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(y_true * y_pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim = ssim_n / ssim_d
    return ssim.mean()

def ssim(y_true, y_pred, y_mask, size_image=HR_SIZE, max_val=65535.0, clear_only=False):
    """
    Modified SSIM metric to take into account pixel shifts and mask, using PyTorch
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_cropped_image = size_image - max_pixels_shifts
    cropped_predictions = y_pred[:, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, i:i + size_cropped_image, j:j + size_cropped_image]
            cropped_y_mask = y_mask[:, i:i + size_cropped_image, j:j + size_cropped_image]

            cropped_y_mask = cropped_y_mask.float()

            cropped_predictions_masked = cropped_predictions.float() * cropped_y_mask
            cropped_labels_masked = cropped_labels.float() * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=[1, 2])

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(
                cropped_labels_masked - cropped_predictions_masked,
                dim=[1, 2]
            )

            b = b.view(-1, 1, 1, 1)

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions *= cropped_y_mask

            # SSIM calculation
            cSSIM = structural_similarity(corrected_cropped_predictions, cropped_labels_masked, data_range=max_val)
            if clear_only:
                cSSIM = (cSSIM - 1) * total_pixels_masked / clear_pixels + 1
            X.append(cSSIM)

    X = torch.stack(X)
    max_cSSIM = torch.max(X, dim=0)[0]
    return torch.mean(max_cSSIM)
