import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
import pandas as pd
import os
from utils.Dataloader import get_inference_dataloader
import warnings
warnings.filterwarnings('ignore')

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, act=True):
        super().__init__()

        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]

        if act == True:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            conv_block(in_c, out_c),
            conv_block(out_c, out_c)
        )
        self.p1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        p = self.p1(x)
        return x, p

class unet3plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.e5 = nn.Sequential(
            conv_block(512, 1024),
            conv_block(1024, 1024)
        )

        """ Decoder 4 """
        self.e1_d4 = conv_block(64, 64)
        self.e2_d4 = conv_block(128, 64)
        self.e3_d4 = conv_block(256, 64)
        self.e4_d4 = conv_block(512, 64)
        self.e5_d4 = conv_block(1024, 64)

        self.d4 = conv_block(64*5, 64)

        """ Decoder 3 """
        self.e1_d3 = conv_block(64, 64)
        self.e2_d3 = conv_block(128, 64)
        self.e3_d3 = conv_block(256, 64)
        self.e4_d3 = conv_block(64, 64)
        self.e5_d3 = conv_block(1024, 64)

        self.d3 = conv_block(64*5, 64)

        """ Decoder 2 """
        self.e1_d2 = conv_block(64, 64)
        self.e2_d2 = conv_block(128, 64)
        self.e3_d2 = conv_block(64, 64)
        self.e4_d2 = conv_block(64, 64)
        self.e5_d2 = conv_block(1024, 64)

        self.d2 = conv_block(64*5, 64)

        """ Decoder 1 """
        self.e1_d1 = conv_block(64, 64)
        self.e2_d1 = conv_block(64, 64)
        self.e3_d1 = conv_block(64, 64)
        self.e4_d1 = conv_block(64, 64)
        self.e5_d1 = conv_block(1024, 64)

        self.d1 = conv_block(64*5, 64)

        """ Output """
        self.y1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, inputs):
        """ Encoder """
        e1, p1 = self.e1(inputs)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)

        """ Bottleneck """
        e5 = self.e5(p4)

        """ Decoder 4 """
        e1_d4 = F.max_pool2d(e1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(e2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(e3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(e4)

        e5_d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True)
        e5_d4 = self.e5_d4(e5_d4)

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)
        d4 = self.d4(d4)

        """ Decoder 3 """
        e1_d3 = F.max_pool2d(e1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)
        d3 = self.d3(d3)

        """ Decoder 2 """
        e1_d2 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)

        d2 = torch.cat([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], dim=1)
        d2 = self.d2(d2)

        """ Decoder 1 """
        e1_d1 = self.e1_d1(e1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)

        d1 = torch.cat([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], dim=1)
        d1 = self.d1(d1)

        """ Output """
        y1 = self.y1(d1)

        return torch.sigmoid(y1)

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
    weight_path = config['Weights']['unet3plus']
    num_workers = config['num_workers']
    inference_path = config['paths']['inference_dir']
    threshold = config['threshold']
    os.makedirs(inference_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet3plus().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model_dir = os.path.join(inference_path, model.__class__.__name__)
    os.makedirs(model_dir, exist_ok=True)
    
    model.eval()
    target_layer = model.y1
    gradcam = GradCAM(model=model, target_layer=target_layer)
    
    dataloader = get_inference_dataloader(
        image_path, mask_path, json_file,
        image_size=(512, 512),
        mask_size=(512, 512),
        batch_size=1,
        inchannel=3,
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