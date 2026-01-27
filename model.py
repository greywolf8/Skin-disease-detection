"""
Deep learning models for ISIC 2016 skin disease classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import logging

from config import NUM_CLASSES, DEVICE


class SkinDiseaseClassifier(nn.Module):
    """Skin disease classifier using transfer learning"""
    
    def __init__(self, model_name='resnet50', num_classes=NUM_CLASSES, pretrained=True):
        super(SkinDiseaseClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'efficientnet-b0':
            if pretrained:
                self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            else:
                self.backbone = EfficientNet.from_name('efficientnet-b0')
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            
        elif model_name == 'efficientnet-b3':
            if pretrained:
                self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
            else:
                self.backbone = EfficientNet.from_name('efficientnet-b3')
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        logging.info(f"Created {model_name} model with {num_classes} classes")
        logging.info(f"Feature dimension: {num_features}")
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logging.info("Backbone parameters frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logging.info("Backbone parameters unfrozen")


def create_model(model_name='resnet50', num_classes=NUM_CLASSES, pretrained=True):
    """Create and return a model instance"""
    
    try:
        model = SkinDiseaseClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
        
        # Move to device
        model = model.to(DEVICE)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"Model created successfully")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Device: {DEVICE}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}")
        raise


def load_model_checkpoint(checkpoint_path, model_name='resnet50', num_classes=NUM_CLASSES):
    """Load model from checkpoint"""
    
    try:
        # Create model
        model = create_model(model_name, num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logging.info(f"Model loaded from {checkpoint_path}")
        
        # Load additional info if available
        epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        
        return model, epoch, best_acc
        
    except Exception as e:
        logging.error(f"Error loading model checkpoint: {str(e)}")
        raise


def save_model_checkpoint(model, optimizer, epoch, best_acc, filepath, 
                         additional_info=None):
    """Save model checkpoint"""
    
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'model_name': model.model_name,
            'num_classes': model.num_classes
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
        
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        raise


def get_model_summary(model, input_size=(3, 224, 224)):
    """Get model summary"""
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(DEVICE)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    summary = {
        'model_name': model.model_name,
        'input_shape': input_size,
        'output_shape': output.shape[1:],
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': count_parameters(model),
        'device': str(next(model.parameters()).device)
    }
    
    return summary
