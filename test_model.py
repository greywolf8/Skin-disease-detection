#!/usr/bin/env python3
"""
Test trained model on test dataset
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import load_model_checkpoint
from dataset import create_test_loader
from utils import setup_logging
import logging

def test_model(model_path, data_dir, metadata_file, batch_size=32, model_name='resnet50'):
    """Test the model on test dataset"""
    
    setup_logging()
    
    print("="*60)
    print("TESTING TRAINED MODEL ON TEST DATASET")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Check if test directory exists
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return
    
    print(f"Loading model from: {model_path}")
    print(f"Test data directory: {test_dir}")
    
    try:
        # Load the trained model
        model, _, _ = load_model_checkpoint(model_path, model_name)
        model.eval()
        print("Model loaded successfully!")
        
        # Create test data loader
        test_loader = create_test_loader(data_dir, metadata_file, batch_size)
        print(f"Test dataset loaded: {len(test_loader.dataset)} images")
        
        # Test the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total test samples: {len(all_labels)}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=['Benign', 'Malignant']))
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Benign  Malignant")
        print(f"Actual Benign    {cm[0,0]:3d}      {cm[0,1]:3d}")
        print(f"    Malignant    {cm[1,0]:3d}      {cm[1,1]:3d}")
        
        # Individual predictions with confidence
        print(f"\nIndividual Predictions:")
        print(f"{'Image':<15} {'Actual':<10} {'Predicted':<10} {'Confidence':<12} {'Correct':<8}")
        print("-" * 60)
        
        # Get image filenames for detailed output
        test_images = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.jpg'):
                    test_images.append(file)
        test_images.sort()
        
        correct_predictions = 0
        for i, (actual, predicted, probs) in enumerate(zip(all_labels, all_predictions, all_probabilities)):
            confidence = probs[predicted]
            is_correct = actual == predicted
            correct_predictions += is_correct
            
            actual_label = 'Benign' if actual == 0 else 'Malignant'
            pred_label = 'Benign' if predicted == 0 else 'Malignant'
            
            image_name = test_images[i] if i < len(test_images) else f"Image_{i+1}"
            
            print(f"{image_name:<15} {actual_label:<10} {pred_label:<10} {confidence:.4f}       {'✓' if is_correct else '✗'}")
        
        print(f"\nSummary:")
        print(f"Correct predictions: {correct_predictions}/{len(all_labels)}")
        print(f"Final test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return None

if __name__ == "__main__":
    # Default parameters
    model_path = "checkpoints/best_model.pth"
    data_dir = "ISIC_2016"
    metadata_file = "ISIC_2016/metadata/ISIC_2016_Training_Metadata.csv"
    
    # Test the model
    accuracy = test_model(model_path, data_dir, metadata_file)
    
    if accuracy is not None:
        print(f"\nTesting completed successfully!")
        print(f"Model can be used for predictions on new images using:")
        print(f"python main.py predict --model-path {model_path} --image path/to/image.jpg")
    else:
        print("Testing failed!")