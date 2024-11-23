import os
import yaml
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import monai
from typing import Dict, Any, List, Tuple
from utils import dataloader, helper, visualisation
from Models import (
    attention_unet,
    UnetASPP,
    deeplabv3plus,
    unet,
    swin_unet,
    segformer,
    unet3plus
)

__author__ = "Sarang Banakhede, Dhruve Kiyawat"
__version__ = "1.0"
__date__ = "2024-08-13"

ModelConfig = Tuple[torch.nn.Module, int, Tuple[int, int], Tuple[int, int]]

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse configuration settings from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")

def get_model_configurations() -> List[ModelConfig]:
    """
    Define and return configurations for all SOTA models.
    
    Returns:
        List of tuples containing (model, input_channels, image_size, mask_size)
    """
    return [
        (attention_unet.attention_unet(), 1, (512, 512), (512, 512)),
        (UnetASPP.UNetASPP(), 1, (512, 512), (512, 512)),
        (deeplabv3plus.Deeplabv3Plus(), 3, (512, 512), (512, 512)),
        (unet.unet(), 3, (512, 512), (512, 512)),
        (unet3plus.unet3plus(), 3, (512, 512), (512, 512)),
        (swin_unet.SwinTransformerSys(), 3, (224, 224), (224, 224)),
        (segformer.get_segformer_model(num_labels=1, image_size=(512, 512)), 3, (512, 512), (128, 128))
    ]

def setup_training_environment(config: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """
    Setup training environment and create necessary directories.
    
    Args:
        config: Configuration dictionary containing path settings

    Returns:
        Tuple of (device, paths_dict) where paths_dict contains all created directories
    """
    paths = {
        'results': config['paths']['result_path'],
        'plots': config['paths']['plot_dir'],
        'intermediate': config['paths']['intermidiate_plot_dir'],
        'weights': config['paths']['weights_dir']
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device, paths

def train_model(
    model_config: ModelConfig,
    config: Dict[str, Any],
    device: str,
    paths: Dict[str, str]
) -> None:
    """
    Train and evaluate a single model configuration.
    
    Args:
        model_config: Tuple containing model specifications
        config: Training configuration parameters
        device: Device to run training on
        paths: Dictionary of paths for saving results
        
    Returns:
        None
    """
    model, channels, img_size, mask_size = model_config
    model.to(device)
    model_name = model.__class__.__name__
    
    writer = SummaryWriter(f'runs/{model_name}')
    
    train_dataloader, test_dataloader = dataloader.get_loader(
        image_dir=config['paths']['img_path'],
        mask_dir=config['paths']['mask_path'],
        batch_size=config['training']['batch_size'],
        image_size=img_size,
        mask_size=mask_size,
        num_workers=config['training']['num_workers'],
        train_split=config['training']['train_split'],
        inchannel=channels
    )
    
    print(f"\nTraining and testing {model_name}...")
    
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = monai.losses.DiceCELoss(to_onehot_y=False, softmax=False, sigmoid=False)
    
    train_results, test_results = helper.fit_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config['training']['epochs'],
        device=device,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        writer=writer,
        weight_dir=paths['weights'],
        save_epoch=config['training']['save_epoch'],
        intermediate_path=paths['intermediate']
    )
    
    helper.save_results(model_name, train_results, test_results, paths['results'])
    writer.close()
    print("-" * 50)

def main() -> None:
    """
    Main execution function for training and evaluating SOTA models for tumor segmentation.
    
    This function:
    1. Loads configuration
    2. Sets up training environment
    3. Trains and evaluates each model
    4. Generates visualization plots
    """
    try:
        config = load_config('config.yaml')
        device, paths = setup_training_environment(config)
        
        model_configs = get_model_configurations()
        for model_config in model_configs:
            train_model(model_config, config, device, paths)
        
        print("\nPlotting all metrics for each model...")
        visualisation.plot(paths['results'], paths['plots'])
        
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()