"""
Description: This file contains the implementation of the Attention U-Net model.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
import os
import pandas as pd
from utils.Dataloader import get_inference_dataloader
import warnings
warnings.filterwarnings('ignore')

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
                                  nn.BatchNorm2d(num_features = out_channels),
                                  nn.ReLU(inplace = True),
                                  nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
                                  nn.BatchNorm2d(num_features = out_channels),
                                  nn.ReLU(inplace = True))
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Block, self).__init__()

        self.conv = Conv_Block(in_channels = in_channels, out_channels = out_channels)
        self.pool = nn.MaxPool2d(kernel_size = (2,2))

    def forward(self, x):
        s = self.conv(x)
        output = self.pool(s)
        return s, output
    
class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = Conv_Block(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = Encoder_Block(1, 64)
        self.e2 = Encoder_Block(64, 128)
        self.e3 = Encoder_Block(128, 256)

        self.b1 = Conv_Block(256, 512)

        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return torch.sigmoid(output)
    
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
    weight_path = config['Weights']['attention_unet']
    num_workers = config['num_workers']
    inference_path = config['paths']['inference_dir']
    threshold = config['threshold']
    os.makedirs(inference_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attention_unet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model_dir = os.path.join(inference_path, model.__class__.__name__)
    os.makedirs(model_dir, exist_ok=True)
    
    model.eval()
    target_layer = model.d3.c1.conv[0] 
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

        for ax, (img, title, cmap) in zip(axes, plots):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        
        data['image_name'].append(name[0])
        data['dice'].append(dice)
        data['iou'].append(iou)
        data['precision'].append(precision)
        data['recall'].append(recall)
        data['f1'].append(f1)
        data['accuracy'].append(accuracy)
                
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