"""
This module contains functions for evaluating the performance of a segmentation model.
It includes metrics such as Dice coefficient, IoU, precision, recall, F1 score, and pixel accuracy.
"""

import torch
import numpy as np
from typing import Dict, List


def calculate_dice(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate the Dice coefficient between predicted and true segmentation masks.
    
    Args:
        y_pred (torch.Tensor): Predicted segmentation mask
        y_true (torch.Tensor): Ground truth segmentation mask
    
    Returns:
        float: Dice coefficient
    """
    smooth = 1e-8  # To avoid division by zero
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def calculate_iou(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate the Intersection over Union (IoU) between predicted and true segmentation masks.
    
    Args:
        y_pred (torch.Tensor): Predicted segmentation mask
        y_true (torch.Tensor): Ground truth segmentation mask
    
    Returns:
        float: IoU score
    """
    smooth = 1e-8  # To avoid division by zero
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_precision_recall_f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple:
    """
    Calculate precision, recall, and F1 score for binary segmentation.
    
    Args:
        y_pred (torch.Tensor): Predicted segmentation mask
        y_true (torch.Tensor): Ground truth segmentation mask
    
    Returns:
        tuple: (precision, recall, F1 score)
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    true_positives = torch.sum((y_pred == 1) & (y_true == 1))
    false_positives = torch.sum((y_pred == 1) & (y_true == 0))
    false_negatives = torch.sum((y_pred == 0) & (y_true == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return (precision.item(), recall.item(), f1.item())

def eval(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the performance of a segmentation model using various metrics.
    
    Args:
        y_pred (np.ndarray): Predicted segmentation masks
        y_true (np.ndarray): Ground truth segmentation masks
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing evaluation metrics for foreground and background
    """
    # Convert numpy arrays to PyTorch tensors
    y_pred_fg = torch.from_numpy(y_pred)
    y_true_fg = torch.from_numpy(y_true)
    y_pred_binary_fg = (y_pred_fg > 0.5).float()

    y_pred_bg = 1 - y_pred_fg
    y_true_bg = 1 - y_true_fg
    y_pred_binary_bg = (y_pred_bg > 0.5).float()

    # Ensure inputs are in the correct shape (B, C, H, W)
    if y_pred_fg.ndim == 3:
        y_pred_fg = y_pred_fg.unsqueeze(1)
        y_pred_binary_fg = y_pred_binary_fg.unsqueeze(1)
    if y_true_fg.ndim == 3:
        y_true_fg = y_true_fg.unsqueeze(1)

    # Initialize lists to store metrics
    metrics = {
        "foreground": {"dice_scores": [], "iou_scores": [], "precision": [], "recall": [], "f1": [], "pixel_accuracies": []},
        "background": {"dice_scores": [], "iou_scores": [], "precision": [], "recall": [], "f1": [], "pixel_accuracies": []}
    }

    # Calculate metrics for each image in the batch
    for i in range(y_pred_binary_fg.shape[0]):
        for cls in ["foreground", "background"]:
            y_pred = y_pred_binary_fg[i] if cls == "foreground" else y_pred_binary_bg[i]
            y_true = y_true_fg[i] if cls == "foreground" else y_true_bg[i]

            precision, recall, f1 = calculate_precision_recall_f1(y_pred, y_true)
            metrics[cls]["precision"].append(precision)
            metrics[cls]["recall"].append(recall)
            metrics[cls]["f1"].append(f1)

            correct_pixels = torch.sum(y_pred == y_true)
            total_pixels = torch.prod(torch.tensor(y_true.shape))
            pixel_accuracy = correct_pixels.float() / total_pixels
            metrics[cls]["pixel_accuracies"].append(pixel_accuracy.item())

            metrics[cls]["dice_scores"].append(calculate_dice(y_pred, y_true))
            metrics[cls]["iou_scores"].append(calculate_iou(y_pred, y_true))

    result = {}
    for cls in ["foreground", "background"]:
        result[cls] = {metric: np.mean(values) for metric, values in metrics[cls].items()}

    return result

if __name__=='__main__':
    pass
