"""
Configuration settings for ISIC 2016 skin disease classification
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing settings
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet means
STD = [0.229, 0.224, 0.225]   # ImageNet stds

# Training settings
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_VAL_SPLIT = 0.2

# Data augmentation settings
ROTATION_DEGREES = 15
BRIGHTNESS_FACTOR = 0.2
CONTRAST_FACTOR = 0.2
SATURATION_FACTOR = 0.2
HUE_FACTOR = 0.1

# Model settings
NUM_CLASSES = 2  # Benign, Malignant
CLASS_NAMES = ['Benign', 'Malignant']

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Checkpoint settings
CHECKPOINT_FREQUENCY = 5  # Save checkpoint every N epochs
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs

# Evaluation settings
CONFIDENCE_THRESHOLD = 0.5
