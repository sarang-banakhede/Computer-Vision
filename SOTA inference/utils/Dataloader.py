import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
import nibabel as nib
import pydicom
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
import tifffile
import cv2


class InferenceDataset(Dataset):
    """A dataset class for loading medical images and masks from various formats.
    
    Supports NIFTI, DICOM, TIFF, and standard image formats.
    Handles both 2D and 3D images with optional slice selection.
    """
    
    def __init__(self, image_dir: str, mask_dir: str, json_file: str, 
                 inchannel: int = 1, img_transform = None, mask_transform = None):
        """Initialize the dataset with directories and transforms.

        Args:
            image_dir: Root directory containing images
            mask_dir: Root directory containing masks
            json_file: Path to JSON file with image/mask filenames
            inchannel: Number of input channels (1 for grayscale, 3 for RGB)
            img_transform: Optional transform for images
            mask_transform: Optional transform for masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.inchannel = inchannel
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        with open(json_file, 'r') as f:
            self.image_list = json.load(f)
        
        self.imgs = self.image_list.get('test_files')
        self.masks = self.image_list.get('test_files')
        self.test_img = self.image_list.get('test_slices')

    def _load_image(self, path: str, slice_idx: int = None) -> np.ndarray:
        """Load an image from the specified path and format.

        Args:
            path: Path to the image file
            slice_idx: Optional index for 3D volume slicing

        Returns:
            Loaded image as numpy array

        Raises:
            ValueError: If file extension is not supported
        """
        ext = os.path.splitext(path)[-1].lower()
        
        if ext in ['.nii', '.gz', '.nii.gz']:
            img_data = nib.load(path).get_fdata()
            return img_data[:,:,slice_idx] if slice_idx is not None else img_data
            
        if ext in ['.dcm', '.dicom']:
            img_data = pydicom.dcmread(path).pixel_array
            return img_data if slice_idx is None else img_data[:, :, slice_idx]
            
        if ext in ['.tif', '.tiff']:
            return tifffile.imread(path)
            
        if ext in ['.png', '.jpg', '.jpeg']:
            return io.imread(path)
            
        raise ValueError(f"Unsupported file extension: {ext}")

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Standardize image format to match specified input channels.

        Args:
            img: Input image array

        Returns:
            Preprocessed image array as float32
        """
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        if self.inchannel == 1:
            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            if img.shape[-1] == 3:
                img = rgb2gray(img)
                img = np.expand_dims(img, axis=-1)
        elif self.inchannel == 3:
            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

        return img.astype(np.float32)

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to binary format with appropriate dimensions.

        Args:
            mask: Input mask array

        Returns:
            Preprocessed binary mask as float32
        """
        if mask.ndim == 3:
            if mask.shape[-1] == 4:
                mask = rgba2rgb(mask)
            if mask.shape[-1] == 3:
                mask = rgb2gray(mask)
                
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def __len__(self) -> int:
        """Return the total number of slices/images in dataset."""
        return len(self.test_img)

    def __getitem__(self, idx: int) -> tuple:
        """Get image, mask, and filename for given index.

        Args:
            idx: Index to retrieve

        Returns:
            Tuple of (image, mask, filename)
        """
        file_idx, slice_idx = self.test_img[idx]
        img_path = os.path.join(self.image_dir, self.imgs[file_idx])
        mask_path = os.path.join(self.mask_dir, self.masks[file_idx])

        img = self._load_image(img_path, slice_idx=slice_idx)
        img = self._preprocess_image(img)
        mask = self._load_image(mask_path, slice_idx=slice_idx)
        mask = self._preprocess_mask(mask)
        
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask, f"{self.imgs[file_idx]}_{slice_idx}"


def binary_mask_transform(mask: np.ndarray, image_size: tuple) -> torch.Tensor:
    """Convert mask to binary tensor at specified size.

    Args:
        mask: Input mask array
        image_size: Target size as (height, width)

    Returns:
        Binary mask tensor
    """
    mask = transforms.ToPILImage()(mask)
    mask = mask.resize(image_size, resample=Image.NEAREST)
    mask = transforms.ToTensor()(mask)
    return (mask > 0).float()


def get_inference_dataloader(
    image_dir: str,
    mask_dir: str,
    json_file: str,
    image_size: tuple = (512, 512),
    mask_size: tuple = (512, 512),
    batch_size: int = 1,
    inchannel: int = 1,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for inference with specified parameters.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        json_file: JSON file with image/mask filenames
        image_size: Target size for images
        mask_size: Target size for masks
        batch_size: Batch size for DataLoader
        inchannel: Number of input channels
        num_workers: Number of worker processes

    Returns:
        Configured DataLoader instance
    """
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size)
    ])
    mask_transform = lambda x: binary_mask_transform(x, mask_size)

    dataset = InferenceDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        json_file=json_file,
        inchannel=inchannel,
        img_transform=img_transform,
        mask_transform=mask_transform
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


if __name__ == '__main__':
    pass