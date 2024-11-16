import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
import PIL.Image as Image
import nibabel as nib
import pydicom
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
import tifffile
import json
import random

class CustomDataset(Dataset):
    """
    CustomDataset for medical images that ensures all slices from a 3D volume stay in same split.
    Supports: NIFTI (.nii, .nii.gz), DICOM (.dcm, .dicom), TIFF (.tif, .tiff), and standard images (.png, .jpg, .jpeg)
    """
    def __init__(self, files, image_dir, mask_dir, inchannel, img_transform=None, mask_transform=None, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.files = files
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.inchannel = inchannel
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.slice_index = []
        self._preprocess_dicom_nifti()

    def _is_valid_file(self, filename):
        valid_extensions = ('.nii', '.nii.gz', '.dcm', '.dicom', '.tif', '.tiff', '.png', '.jpg', '.jpeg')
        return filename.lower().endswith(valid_extensions)

    def _preprocess_dicom_nifti(self):
        for idx, img_file in enumerate(self.files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, img_file) if self.mask_dir else None
            ext = '.' + '.'.join(img_file.split('.')[1:])
            
            img_data, _ = self._load_image(img_path)
            if mask_path:
                mask_data, _ = self._load_image(mask_path)
            else:
                mask_data = np.zeros_like(img_data)
            
            if ext in ['.dcm', '.dicom', '.nii', '.nii.gz']:
                num_slices = img_data.shape[-1]
                for slice_idx in range(num_slices):
                    mask_slice = np.squeeze(mask_data[:, :, slice_idx])
                    if np.any(mask_slice > 0):
                        self.slice_index.append((idx, slice_idx))
            else:
                if np.any(mask_data > 0):
                    self.slice_index.append((idx, None))

    def _load_image(self, path):
        ext = '.' + '.'.join(path.split('.')[1:])
        if ext in ['.nii', '.nii.gz']:
            return nib.load(path).get_fdata(), ext
        elif ext in ['.dcm', '.dicom']:
            return pydicom.dcmread(path).pixel_array, ext
        elif ext in ['.tif', '.tiff']:
            return io.imread(path), ext  
        elif ext in ['.png', '.jpg', '.jpeg']:
            return io.imread(path), ext
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _preprocess_image(self, img):
        
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
    
    def _preprocess_mask(self, mask):
        if mask.ndim == 3:
            if mask.shape[-1] == 4:  
                mask = rgba2rgb(mask)
            if mask.shape[-1] == 3:  
                mask = rgb2gray(mask)

        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_index[idx]
        img_path = os.path.join(self.image_dir, self.files[file_idx])
        mask_path = os.path.join(self.mask_dir, self.files[file_idx]) if self.mask_dir else None
        img, ext = self._load_image(img_path)
        if mask_path:
            mask, _ = self._load_image(mask_path)
        else:
            mask = np.zeros_like(img)
        
        if ext in ['.dcm', '.dicom', '.nii', '.nii.gz']:
            img = np.squeeze(img[:, :, slice_idx])
            mask = np.squeeze(mask[:, :, slice_idx])
        img = self._preprocess_image(img)
        mask = self._preprocess_mask(mask)
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask

def binary_mask_transform(mask, image_size):
    mask = transforms.ToPILImage()(mask)
    mask = mask.resize(image_size, resample=Image.NEAREST)
    mask = transforms.ToTensor()(mask)
    mask = (mask > 0).float()
    return mask

def get_split_loaders(image_dir, mask_dir, image_size=(512, 512), mask_size=(512, 512), 
                     batch_size=1, num_workers=0, train_split=0.5, inchannel=1, 
                     save_splits=True, seed=42):
    """
    Creates data loaders with image-level splitting, ensuring all slices from one 3D volume
    remain in either train or test set.
    
    Returns:
        tuple: (train_loader, test_loader, split_info)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: List All Valid Files in Directory
    valid_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(
        ('.nii', '.nii.gz', '.dcm', '.dicom', '.tif', '.tiff', '.png', '.jpg', '.jpeg'))])

    # Step 2: Split the List into Train and Test
    n_files = len(valid_files)
    n_train = int(n_files * train_split)
    shuffled_files = valid_files.copy()
    random.shuffle(shuffled_files)
    
    train_files = shuffled_files[:n_train]
    test_files = shuffled_files[n_train:]

    # Step 3: Set Up Transforms
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    mask_transform = lambda x: binary_mask_transform(x, mask_size)

    # Step 4: Create Train and Test Datasets
    train_dataset = CustomDataset(train_files, image_dir, mask_dir, inchannel,
                                  img_transform=img_transform,
                                  mask_transform=mask_transform)

    test_dataset = CustomDataset(test_files, image_dir, mask_dir, inchannel,
                                 img_transform=img_transform,
                                 mask_transform=mask_transform)

    # Step 5: Create DataLoaders for Train and Test Sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # Step 6: Save Split Information if Required
    split_info = {
        'train_files': train_files,
        'test_files': test_files,
        'train_slices': train_dataset.slice_index,
        'test_slices': test_dataset.slice_index
    }

    if save_splits:
        with open('Split_info.json', 'w') as f:
            json.dump(split_info, f, indent=4)

    return train_loader, test_loader

if __name__ == '__main__':
    pass