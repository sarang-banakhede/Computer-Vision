import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from .evaluate import eval, calculate_dice, calculate_iou
from PIL import Image
import random

def print_results(phase: str, results: Dict[str, Any]) -> None:
    """
    Print evaluation metrics for a given phase.
    
    Args:
        phase: The evaluation phase ('Train' or 'Test')
        results: Dictionary containing evaluation metrics including loss, dice scores, IoU, F1, precision, recall, and pixel accuracies
    """
    print(f"{phase} loss: {results['loss']:.4f}")
    for class_type in ['foreground', 'background']:
        print(f"{class_type.capitalize()} - "
              f"Dice: {results[class_type]['dice_scores']:.4f}, "
              f"IoU: {results[class_type]['iou_scores']:.4f}, "
              f"F1: {results[class_type]['f1']:.4f}, "
              f"Precision: {results[class_type]['precision']:.4f}, "
              f"Recall: {results[class_type]['recall']:.4f}, "
              f"Pixel Accuracy: {results[class_type]['pixel_accuracies']:.4f}")

def extract_logits_from_output(output: Any) -> torch.Tensor:
    """
    Extract logits from various possible model output formats.
    
    Args:
        output: Model output which may be a tensor, object with logits attribute, or dictionary

    Returns:
        Tensor containing the logits

    Raises:
        ValueError: If output format is not supported
    """
    if hasattr(output, 'logits'):
        return output.logits
    if isinstance(output, dict) and 'logits' in output:
        return output['logits']
    if isinstance(output, torch.Tensor):
        return output
    raise ValueError("Unsupported model output format")

def train_epoch(model: nn.Module, optimizer: Adam, loss_fn: nn.Module, device: str, dataloader: DataLoader) -> Dict[str, Any]:
    """
    Execute one training epoch.
    
    Args:
        model: Neural network model to train
        optimizer: Optimization algorithm
        loss_fn: Loss function for training
        device: Device to run computations on
        dataloader: DataLoader providing training batches

    Returns:
        Dictionary containing training metrics for the epoch
    """
    model.train()
    label_list, logits_list = [], []
    total_loss = 0.0
    
    for image, labels in tqdm(dataloader):
        image, labels = image.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logits = extract_logits_from_output(model(image))
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        label_list.append(labels.cpu().numpy())
        logits_list.append(logits.cpu().detach().numpy())
        total_loss += loss.item()

    results = eval(np.concatenate(logits_list, axis=0), np.concatenate(label_list, axis=0))
    results['loss'] = total_loss / len(dataloader)
    print_results("Train", results)
    return results

def test_epoch(model: nn.Module, loss_fn: nn.Module, device: str, dataloader: DataLoader) -> Dict[str, Any]:
    """
    Execute one testing epoch.
    
    Args:
        model: Neural network model to evaluate
        loss_fn: Loss function for evaluation
        device: Device to run computations on
        dataloader: DataLoader providing test batches

    Returns:
        Dictionary containing evaluation metrics for the epoch
    """
    model.eval()
    label_list, logits_list = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for image, labels in tqdm(dataloader):
            image, labels = image.to(device), labels.to(device)
            logits = extract_logits_from_output(model(image))
            loss = loss_fn(logits, labels)
            
            label_list.append(labels.cpu().numpy())
            logits_list.append(logits.cpu().detach().numpy())
            total_loss += loss.item()

    results = eval(np.concatenate(logits_list, axis=0), np.concatenate(label_list, axis=0))
    results['loss'] = total_loss / len(dataloader)
    print_results("Test", results)
    return results

def save_intermediate_results(model: nn.Module, dataloader: DataLoader, intermediate_path: str, 
                            mode: str, device: str, epoch: int) -> np.ndarray:
    """
    Save and visualize intermediate predictions during training.
    
    Args:
        model: Neural network model
        dataloader: DataLoader providing data samples
        intermediate_path: Directory to save visualization results
        mode: Evaluation mode ('train' or 'test')
        device: Device to run computations on
        epoch: Current training epoch

    Returns:
        Numpy array containing the visualization image
    """
    model.eval()
    with torch.no_grad():
        image, mask = next(iter(dataloader))
        idx = random.randint(0, image.size(0) - 1)
        
        image = image[idx].to(device)
        mask = mask[idx].to(device)
        output = model(image.unsqueeze(0))
        
        if isinstance(output, dict):
            prediction = output['logits']
        else:
            prediction = output           
        
        image = image.permute(1, 2, 0).cpu().detach()
        mask = mask.permute(1, 2, 0).cpu().detach()
        prediction = prediction.squeeze(0).permute(1, 2, 0).cpu().detach()
        
        dice_score = calculate_dice(prediction > 0.5, mask)
        iou_score = calculate_iou(prediction > 0.5, mask)
        
        plt.figure(figsize=(20, 5))
        plt.suptitle(f'Dice: {dice_score:.8f}, IoU: {iou_score:.8f}', fontsize=16)
        
        subplots = [(1, "Image", image), (2, "Ground Truth", mask),
                    (3, "Prediction", prediction), (4, "Thresholded Prediction", prediction > 0.5)]
        
        for i, title, data in subplots:
            plt.subplot(1, 4, i)
            plt.imshow(data.numpy())
            plt.title(title)
            plt.axis('off')
        
        save_path = os.path.join(intermediate_path, model.__class__.__name__, mode)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'Epoch_{epoch + 1}.png'))
        plt.close()
        
        return np.transpose(np.array(Image.open(os.path.join(save_path, f'Epoch_{epoch + 1}.png'))), (2, 0, 1))

def log_metrics(writer: SummaryWriter, phase: str, metrics: Dict[str, Any], epoch: int) -> None:
    """
    Log training/testing metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        phase: Logging phase ('train' or 'test')
        metrics: Dictionary containing metrics to log
        epoch: Current training epoch
    """
    writer.add_scalar(f'{phase}_loss', metrics['loss'], epoch)
    for class_type in ['foreground', 'background']:
        for metric in ['dice_scores', 'iou_scores', 'f1', 'precision', 'recall', 'pixel_accuracies']:
            writer.add_scalar(f'{phase}_{class_type}_{metric}', metrics[class_type][metric], epoch)

def fit_model(model: nn.Module, optimizer: Adam, loss_fn: nn.Module, epochs: int, device: str,
              train_loader: DataLoader, test_loader: DataLoader, writer: SummaryWriter,
              weight_dir: str, save_epoch: int, intermediate_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate a model for multiple epochs.
    
    Args:
        model: Neural network model to train
        optimizer: Optimization algorithm
        loss_fn: Loss function for training
        epochs: Number of training epochs
        device: Device to run computations on
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        writer: TensorBoard SummaryWriter instance
        weight_dir: Directory to save model weights
        save_epoch: Frequency of saving model weights
        intermediate_path: Directory to save intermediate results

    Returns:
        Tuple containing dictionaries of training and testing metrics for each epoch
    """
    train_metrics, test_metrics = {}, {}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        print("Training:")
        train_epoch_metrics = train_epoch(model, optimizer, loss_fn, device, train_loader)
        log_metrics(writer, 'train', train_epoch_metrics, epoch)
        
        print("\nTesting:")
        test_epoch_metrics = test_epoch(model, loss_fn, device, test_loader)
        log_metrics(writer, 'test', test_epoch_metrics, epoch)
        
        train_metrics[f'epoch_{epoch}'] = train_epoch_metrics
        test_metrics[f'epoch_{epoch}'] = test_epoch_metrics
        
        if (epoch + 1) % save_epoch == 0:
            save_model(model, model.__class__.__name__, weight_dir, epoch + 1)
        
        test_image = save_intermediate_results(model, test_loader, intermediate_path, "test", device, epoch)
        train_image = save_intermediate_results(model, train_loader, intermediate_path, "train", device, epoch)
        writer.add_image('Test Image', test_image, epoch)
        writer.add_image('Train Image', train_image, epoch)
    
    return train_metrics, test_metrics

def save_model(model: nn.Module, model_name: str, model_path: str, epoch: int) -> None:
    """
    Save model weights to disk.
    
    Args:
        model: Neural network model
        model_name: Name of the model
        model_path: Directory to save model weights
        epoch: Current epoch number
    """
    save_path = os.path.join(model_path, model_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pth'))
    print(f"Model saved to {save_path}")

def save_results(model_name: str, train_results: Dict, test_results: Dict, result_metrics_path: str) -> None:
    """
    Save training and testing metrics to JSON files.
    
    Args:
        model_name: Name of the model
        train_results: Dictionary containing training metrics
        test_results: Dictionary containing testing metrics
        result_metrics_path: Directory to save metric files
    """
    model_path = os.path.join(result_metrics_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    for name, results in [("train", train_results), ("test", test_results)]:
        with open(os.path.join(model_path, f'{name}_results.json'), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results for {model_name} saved.")

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

if __name__ == "__main__":
    pass
