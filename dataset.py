"""
Dataset loading and preprocessing for ISIC 2016 skin disease classification
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

from config import *


class ISICDataset(Dataset):
    """ISIC 2016 Dataset for skin lesion classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to images
            labels (list): List of labels (0: Benign, 1: Malignant)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return a black image as fallback
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_metadata(metadata_path):
    """Load and process metadata CSV file"""
    try:
        df = pd.read_csv(metadata_path)
        logging.info(f"Loaded metadata with {len(df)} entries")
        
        # Check required columns
        required_cols = ['isic_id', 'diagnosis_1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and process diagnosis labels
        df['diagnosis_1'] = df['diagnosis_1'].fillna('Unknown')
        
        # Create binary labels: 0 for Benign, 1 for Malignant
        def map_diagnosis(diagnosis):
            if pd.isna(diagnosis) or diagnosis == 'Unknown':
                return None
            diagnosis = str(diagnosis).lower()
            # Handle exact matches first (for our custom labels)
            if diagnosis == 'malignant':
                return 1
            elif diagnosis == 'benign':
                return 0
            # Handle partial matches (for original ISIC labels)
            elif 'malignant' in diagnosis or 'melanoma' in diagnosis:
                return 1
            elif 'benign' in diagnosis:
                return 0
            else:
                return None
        
        df['binary_label'] = df['diagnosis_1'].apply(map_diagnosis)
        
        # Remove rows with unknown diagnoses
        df_clean = df.dropna(subset=['binary_label'])
        df_clean['binary_label'] = df_clean['binary_label'].astype(int)
        
        logging.info(f"After cleaning: {len(df_clean)} entries")
        
        # Print class distribution
        class_counts = df_clean['binary_label'].value_counts()
        logging.info(f"Class distribution:")
        logging.info(f"  Benign (0): {class_counts.get(0, 0)}")
        logging.info(f"  Malignant (1): {class_counts.get(1, 0)}")
        
        return df_clean
        
    except Exception as e:
        logging.error(f"Error loading metadata: {str(e)}")
        raise


def get_image_paths(data_dir, isic_ids, split='train'):
    """Get full image paths from ISIC IDs"""
    image_dir = Path(data_dir) / split
    image_paths = []
    valid_ids = []
    valid_labels = []
    
    for i, isic_id in enumerate(isic_ids):
        # Try different extensions
        found = False
        for ext in IMAGE_EXTENSIONS:
            img_path = image_dir / f"{isic_id}{ext}"
            if img_path.exists():
                image_paths.append(str(img_path))
                valid_ids.append(isic_id)
                found = True
                break
        
        if not found:
            logging.warning(f"Image not found for ID: {isic_id}")
    
    return image_paths, valid_ids


def get_transforms(split='train'):
    """Get data transforms for training or validation"""
    if split == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=BRIGHTNESS_FACTOR,
                contrast=CONTRAST_FACTOR,
                saturation=SATURATION_FACTOR,
                hue=HUE_FACTOR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    return transform


def create_data_loaders(data_dir, metadata_file, batch_size=DEFAULT_BATCH_SIZE, 
                       val_split=DEFAULT_VAL_SPLIT, num_workers=4):
    """Create training and validation data loaders"""
    
    # Load metadata
    df = load_metadata(metadata_file)
    
    # Get image paths
    image_paths, valid_ids = get_image_paths(data_dir, df['isic_id'].tolist(), 'train')
    
    # Filter dataframe to only include images that exist
    df_filtered = df[df['isic_id'].isin(valid_ids)].copy()
    
    # Align with image paths order
    id_to_label = dict(zip(df_filtered['isic_id'], df_filtered['binary_label']))
    labels = [id_to_label[isic_id] for isic_id in valid_ids]
    
    logging.info(f"Found {len(image_paths)} valid images")
    
    # Split data
    if val_split > 0:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=val_split, random_state=42, 
            stratify=labels
        )
    else:
        train_paths, train_labels = image_paths, labels
        val_paths, val_labels = [], []
    
    logging.info(f"Training set: {len(train_paths)} images")
    logging.info(f"Validation set: {len(val_paths)} images")
    
    # Print class distributions
    train_counter = Counter(train_labels)
    logging.info(f"Training class distribution: {dict(train_counter)}")
    
    if val_labels:
        val_counter = Counter(val_labels)
        logging.info(f"Validation class distribution: {dict(val_counter)}")
    
    # Create datasets
    train_dataset = ISICDataset(
        train_paths, train_labels, 
        transform=get_transforms('train')
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = None
    if val_paths:
        val_dataset = ISICDataset(
            val_paths, val_labels,
            transform=get_transforms('val')
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    # Calculate class weights for weighted loss
    class_weights = None
    if len(train_counter) == 2:  # Binary classification
        total = sum(train_counter.values())
        class_weights = torch.FloatTensor([
            total / (2 * train_counter[0]),  # Weight for class 0 (Benign)
            total / (2 * train_counter[1])   # Weight for class 1 (Malignant)
        ])
        logging.info(f"Class weights: {class_weights.tolist()}")
    
    return train_loader, val_loader, class_weights


def create_test_loader(data_dir, metadata_file, batch_size=DEFAULT_BATCH_SIZE, num_workers=4):
    """Create test data loader"""
    
    # Load metadata
    df = load_metadata(metadata_file)
    
    # Try both train and test directories for images
    image_paths = []
    valid_ids = []
    
    for split in ['test', 'train']:  # Try test first, then train as fallback
        img_paths, v_ids = get_image_paths(data_dir, df['isic_id'].tolist(), split)
        if img_paths:
            image_paths = img_paths
            valid_ids = v_ids
            logging.info(f"Using images from {split} directory")
            break
    
    if not image_paths:
        raise ValueError("No valid images found in test or train directories")
    
    # Filter dataframe
    df_filtered = df[df['isic_id'].isin(valid_ids)].copy()
    
    # Get labels
    id_to_label = dict(zip(df_filtered['isic_id'], df_filtered['binary_label']))
    labels = [id_to_label[isic_id] for isic_id in valid_ids]
    
    # Create dataset
    test_dataset = ISICDataset(
        image_paths, labels,
        transform=get_transforms('test')
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    logging.info(f"Test set: {len(image_paths)} images")
    
    return test_loader, valid_ids, labels
