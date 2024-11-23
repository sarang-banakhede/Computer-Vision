"""
Description: This file contains the implementation of the UNetASPP model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import cv2
import os
from utils.Dataloader import get_inference_dataloader
import warnings
warnings.filterwarnings('ignore')

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super(ASPP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3x3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv3x3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.conv3x3_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.conv1x1_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.size()
        pool = self.avg_pool(x)
        pool = self.conv1x1_pool(pool)
        pool = self.bn(pool)
        pool = self.relu(F.interpolate(pool, size=(size[2], size[3]), mode='bilinear', align_corners=True))

        conv1x1_1 = self.conv1x1_1(x)
        conv1x1_1 = self.bn(conv1x1_1)
        conv1x1_1 = self.relu(conv1x1_1)

        conv3x3_6 = self.conv3x3_6(x)
        conv3x3_6 = self.bn(conv3x3_6)
        conv3x3_6 = self.relu(conv3x3_6)

        conv3x3_12 = self.conv3x3_12(x)
        conv3x3_12 = self.bn(conv3x3_12)
        conv3x3_12 = self.relu(conv3x3_12)

        conv3x3_18 = self.conv3x3_18(x)
        conv3x3_18 = self.bn(conv3x3_18)
        conv3x3_18 = self.relu(conv3x3_18)

        out = torch.cat([pool, conv1x1_1, conv3x3_6, conv3x3_12, conv3x3_18], dim=1)
        out = self.conv1x1_out(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class UNetASPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetASPP, self).__init__()

        # Encoder (contracting path)
        self.encoder1 = self.contract_block(in_channels, 64, 2)
        self.encoder2 = self.contract_block(64, 128, 2)
        self.encoder3 = self.contract_block(128, 256, 2)
        self.encoder4 = self.contract_block(256, 512, 2)

        # ASPP module
        self.aspp = ASPP(512)

        # Decoder (expanding path)
        self.decoder1 = self.expand_block(512, 256, 2)
        self.decoder2 = self.expand_block(512, 128, 2)
        self.decoder3 = self.expand_block(256, 64, 2)
        self.decoder4 = self.expand_block(128, 32, 2)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels, pool_size=2):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )
        return block

    def expand_block(self, in_channels, out_channels, upsample_scale=2):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upsample_scale, stride=upsample_scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # ASPP
        aspp_out = self.aspp(enc4)
        
        # Decoder
        dec1 = self.decoder1(aspp_out)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec3 = self.decoder3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec4 = self.decoder4(dec3)

        # Final convolution
        final_output = self.final_conv(dec4)
        return torch.sigmoid(final_output)

class GradCAM:
    """Class for generating Gradient Class Activation Maps."""
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image):

        output = self.model(input_image)
        class_loss = torch.mean(output)
        
        self.model.zero_grad()
        class_loss.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  
        return cam

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

def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    data = {
    "image_name": [],
    "dice": [],
    "iou": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "accuracy": []}  
    
    df_path = config['paths']['statistical_result']
    image_path = config['paths']['img_path']
    mask_path = config['paths']['mask_path']
    json_file = config['paths']['json_file']
    weight_path = config['Weights']['unetaspp']
    num_workers = config['num_workers']
    inference_path = config['paths']['inference_dir']
    threshold = config['threshold']
    os.makedirs(inference_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetASPP().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model_dir = os.path.join(inference_path, model.__class__.__name__)
    os.makedirs(model_dir, exist_ok=True)
    
    model.eval()
    target_layer = model.final_conv
    gradcam = GradCAM(model=model, target_layer=target_layer)
    
    dataloader = get_inference_dataloader(
        image_path, mask_path, json_file,
        image_size=(512, 512),
        mask_size=(512, 512),
        batch_size=1,
        inchannel=1,
        num_workers=num_workers
    )
    
    for i, (image, mask, name) in enumerate(dataloader):
        image = image.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            prediction = model(image)
        
        cam = gradcam.generate_cam(image)

        image = image[0].cpu().numpy() 
        mask = mask[0].cpu().numpy().squeeze() 
        prediction = prediction[0].cpu().numpy().squeeze()  
        prediction_binary = (prediction > threshold).astype(np.uint8)

        if image.shape[0] == 1:  
            image = image.squeeze()  
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)  
            image_np = (image * 255).astype(np.uint8)
            image_np = np.stack([image_np] * 3, axis=-1) 
        elif image.shape[0] == 3:  
            image = np.transpose(image, (1, 2, 0))  
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)  
            image_np = (image * 255).astype(np.uint8)
        
        mask_np = (mask * 255).astype(np.uint8)  
        prediction_np = (prediction * 255).astype(np.uint8) 

        prediction_tensor = torch.from_numpy(prediction_binary).float()
        mask_tensor = torch.from_numpy((mask > 0.5).astype(np.uint8)).float()

        dice = calculate_dice(prediction_tensor, mask_tensor)
        iou = calculate_iou(prediction_tensor, mask_tensor)
        precision, recall, f1 = calculate_precision_recall_f1(prediction_tensor, mask_tensor)
        accuracy = (torch.sum(prediction_tensor == mask_tensor) / mask_tensor.numel()).item()

        overlay = np.zeros_like(image_np, dtype=np.uint8)
        overlay[..., 2] = mask_np 
        overlay[..., 1] = prediction_binary * 255 
        overlay_image = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        fig.suptitle(f'Metrics - Dice: {dice:.4f}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}', fontsize=16)

        plots = [
            (image_np, 'Original Image', None),
            (mask_np, 'Ground Truth Mask', 'gray'),
            (prediction_np, 'Predicted Mask', 'gray'),
            (prediction_binary * 255, 'Predicted Mask (Binary)', 'gray'),
            (overlay_image, 'Overlay (Predicted & Groundtruth)', None),
            (cam, 'GradCAM', 'jet'),
        ]
        
        data['image_name'].append(name[0])
        data['dice'].append(dice)
        data['iou'].append(iou)
        data['precision'].append(precision)
        data['recall'].append(recall)
        data['f1'].append(f1)
        data['accuracy'].append(accuracy)
        
        for ax, (img, title, cmap) in zip(axes, plots):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')

        save_path = os.path.join(model_dir, f'{name[0]}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    df = pd.DataFrame(data)
    df_path = os.path.join(df_path,model.__class__.__name__)
    os.makedirs(df_path,exist_ok=True)
    df_path = os.path.join(df_path,'data.csv')
    df.to_csv(df_path, index=False) 
    
    print(f"Inference Completed for {model.__class__.__name__}")

if __name__ == "__main__":
    main()