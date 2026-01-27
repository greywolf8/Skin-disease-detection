"""
Training script for ISIC 2016 skin disease classification
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import json

from model import create_model, save_model_checkpoint, load_model_checkpoint
from dataset import create_data_loaders
from config import DEVICE, CHECKPOINT_FREQUENCY, EARLY_STOPPING_PATIENCE


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    if val_accs:
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def train_model(data_dir, metadata_file, model_name='resnet50', epochs=50, 
                batch_size=32, learning_rate=0.001, val_split=0.2, 
                output_dir='checkpoints', resume_checkpoint=None, 
                use_weighted_loss=False):
    """Train the skin disease classification model"""
    
    logging.info("="*50)
    logging.info("STARTING TRAINING")
    logging.info("="*50)
    
    # Create data loaders
    logging.info("Creating data loaders...")
    train_loader, val_loader, class_weights = create_data_loaders(
        data_dir=data_dir,
        metadata_file=metadata_file,
        batch_size=batch_size,
        val_split=val_split
    )
    
    # Create model
    logging.info(f"Creating {model_name} model...")
    model = create_model(model_name=model_name, pretrained=True)
    
    # Loss function
    if use_weighted_loss and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        logging.info("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        logging.info("Using standard CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    
    if resume_checkpoint:
        logging.info(f"Resuming from checkpoint: {resume_checkpoint}")
        try:
            model, start_epoch, best_acc = load_model_checkpoint(
                resume_checkpoint, model_name
            )
            logging.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.4f}")
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {str(e)}")
            logging.info("Starting training from scratch")
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    logging.info(f"Starting training for {epochs} epochs...")
    logging.info(f"Device: {DEVICE}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        logging.info("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Check for best model
            if val_acc > best_acc:
                best_acc = val_acc
                # Save best model
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                save_model_checkpoint(
                    model, optimizer, epoch+1, best_acc, best_model_path,
                    {'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc}
                )
                logging.info(f"New best model saved with accuracy: {best_acc:.4f}")
            
            # Early stopping check
            early_stopping(val_acc)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered")
                break
        
        else:
            # No validation set, save based on training accuracy
            if train_acc > best_acc:
                best_acc = train_acc
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                save_model_checkpoint(
                    model, optimizer, epoch+1, best_acc, best_model_path,
                    {'train_loss': train_loss, 'train_acc': train_acc}
                )
        
        # Save checkpoint periodically
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_model_checkpoint(
                model, optimizer, epoch+1, best_acc, checkpoint_path
            )
    
    # Training completed
    total_time = time.time() - start_time
    logging.info(f"\nTraining completed in {total_time/3600:.2f} hours")
    logging.info(f"Best accuracy: {best_acc:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    save_model_checkpoint(
        model, optimizer, epoch+1, best_acc, final_model_path
    )
    
    # Plot training history
    if len(train_losses) > 1:
        plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir)
        logging.info("Training plots saved")
    
    # Save training summary
    summary = {
        'model_name': model_name,
        'epochs_trained': len(train_losses),
        'best_accuracy': best_acc,
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'training_time_hours': total_time / 3600,
        'parameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_split': val_split,
            'use_weighted_loss': use_weighted_loss
        }
    }
    
    if val_losses:
        summary['final_val_loss'] = val_losses[-1]
        summary['final_val_acc'] = val_accs[-1]
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Training summary saved to {summary_path}")
    logging.info("Training completed successfully!")
    
    return model, best_acc
