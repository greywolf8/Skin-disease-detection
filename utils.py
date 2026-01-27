"""
Utility functions for ISIC 2016 skin disease classification
"""

import os
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import load_model_checkpoint
from config import DEVICE, IMAGE_SIZE, MEAN, STD, CLASS_NAMES


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('skin_disease_classification.log')
        ]
    )


def get_prediction_transform():
    """Get transform for single image prediction"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def predict_single_image(model_path, image_path, model_name='resnet50'):
    """Predict class for a single image"""
    
    try:
        # Load model
        model, _, _ = load_model_checkpoint(model_path, model_name)
        model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = get_prediction_transform()
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get class probabilities
            probs = probabilities.cpu().numpy()[0]
            
            result = {
                'prediction': CLASS_NAMES[predicted.item()],
                'confidence': confidence.item(),
                'prob_benign': probs[0],
                'prob_malignant': probs[1] if len(probs) > 1 else 0.0
            }
            
            return result
            
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise


def batch_predict_images(model_path, image_paths, model_name='resnet50', batch_size=32):
    """Predict classes for multiple images"""
    
    try:
        # Load model
        model, _, _ = load_model_checkpoint(model_path, model_name)
        model.eval()
        
        transform = get_prediction_transform()
        results = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    logging.warning(f"Could not load image {img_path}: {str(e)}")
                    # Add dummy tensor for failed images
                    batch_images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
            
            if not batch_images:
                continue
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(DEVICE)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                
                # Process results
                for j, (img_path, pred, conf, probs) in enumerate(
                    zip(batch_paths, predictions, confidences, probabilities)
                ):
                    probs_np = probs.cpu().numpy()
                    result = {
                        'image_path': img_path,
                        'prediction': CLASS_NAMES[pred.item()],
                        'confidence': conf.item(),
                        'prob_benign': probs_np[0],
                        'prob_malignant': probs_np[1] if len(probs_np) > 1 else 0.0
                    }
                    results.append(result)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in batch prediction: {str(e)}")
        raise


def calculate_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    from collections import Counter
    import torch
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(0.0)
    
    return torch.FloatTensor(weights)


def get_image_stats(image_paths, sample_size=1000):
    """Calculate dataset statistics for normalization"""
    
    import random
    
    # Sample random images if dataset is large
    if len(image_paths) > sample_size:
        sampled_paths = random.sample(image_paths, sample_size)
    else:
        sampled_paths = image_paths
    
    # Basic transform to resize images
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    pixel_sum = torch.zeros(3)
    pixel_sum_sq = torch.zeros(3)
    num_pixels = 0
    
    logging.info(f"Calculating statistics from {len(sampled_paths)} images...")
    
    for img_path in sampled_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            
            pixel_sum += image_tensor.sum(dim=[1, 2])
            pixel_sum_sq += (image_tensor ** 2).sum(dim=[1, 2])
            num_pixels += image_tensor.shape[1] * image_tensor.shape[2]
            
        except Exception as e:
            logging.warning(f"Could not process {img_path}: {str(e)}")
            continue
    
    # Calculate mean and std
    mean = pixel_sum / num_pixels
    std = torch.sqrt((pixel_sum_sq / num_pixels) - (mean ** 2))
    
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'num_images_processed': len(sampled_paths),
        'num_pixels_total': num_pixels
    }
    
    logging.info(f"Dataset statistics: Mean={mean.tolist()}, Std={std.tolist()}")
    
    return stats


def create_prediction_report(results, output_path):
    """Create a formatted prediction report"""
    
    import pandas as pd
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    summary = {
        'total_predictions': len(results),
        'benign_predictions': len(df[df['prediction'] == 'Benign']),
        'malignant_predictions': len(df[df['prediction'] == 'Malignant']),
        'mean_confidence': df['confidence'].mean(),
        'std_confidence': df['confidence'].std(),
        'min_confidence': df['confidence'].min(),
        'max_confidence': df['confidence'].max()
    }
    
    # Create report
    report = f"""
Skin Disease Classification Prediction Report
{'='*50}

Summary Statistics:
- Total Predictions: {summary['total_predictions']}
- Benign Predictions: {summary['benign_predictions']} ({100*summary['benign_predictions']/summary['total_predictions']:.1f}%)
- Malignant Predictions: {summary['malignant_predictions']} ({100*summary['malignant_predictions']/summary['total_predictions']:.1f}%)

Confidence Statistics:
- Mean Confidence: {summary['mean_confidence']:.4f}
- Std Confidence: {summary['std_confidence']:.4f}
- Min Confidence: {summary['min_confidence']:.4f}
- Max Confidence: {summary['max_confidence']:.4f}

Detailed Predictions:
{'='*50}
"""
    
    for i, result in enumerate(results[:10]):  # Show first 10 predictions
        report += f"""
{i+1}. {os.path.basename(result['image_path'])}
   Prediction: {result['prediction']}
   Confidence: {result['confidence']:.4f}
   Prob(Benign): {result['prob_benign']:.4f}
   Prob(Malignant): {result['prob_malignant']:.4f}
"""
    
    if len(results) > 10:
        report += f"\n... and {len(results) - 10} more predictions\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    # Save detailed CSV
    csv_path = output_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    
    logging.info(f"Prediction report saved to {output_path}")
    logging.info(f"Detailed predictions saved to {csv_path}")
    
    return summary


def validate_dataset_structure(data_dir):
    """Validate the dataset directory structure"""
    
    data_path = os.path.abspath(data_dir)
    
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset directory does not exist: {data_path}")
    
    # Check for required subdirectories
    train_dir = os.path.join(data_path, 'train')
    metadata_dir = os.path.join(data_path, 'metadata')
    
    issues = []
    
    if not os.path.exists(train_dir):
        issues.append(f"Missing train directory: {train_dir}")
    
    if not os.path.exists(metadata_dir):
        issues.append(f"Missing metadata directory: {metadata_dir}")
    
    # Check for metadata file
    metadata_file = os.path.join(metadata_dir, 'ISIC_2016_Training_Metadata.csv')
    if not os.path.exists(metadata_file):
        # Try alternative names
        alt_names = [
            'training_metadata.csv',
            'metadata.csv',
            'ISIC_2016_training_metadata.csv'
        ]
        found = False
        for alt_name in alt_names:
            alt_path = os.path.join(metadata_dir, alt_name)
            if os.path.exists(alt_path):
                metadata_file = alt_path
                found = True
                break
        
        if not found:
            issues.append(f"Metadata CSV file not found in {metadata_dir}")
    
    # Count images in train directory
    if os.path.exists(train_dir):
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend([f for f in os.listdir(train_dir) if f.lower().endswith(ext)])
        
        if len(image_files) == 0:
            issues.append(f"No image files found in {train_dir}")
        else:
            logging.info(f"Found {len(image_files)} image files in training directory")
    
    if issues:
        raise ValueError(f"Dataset validation failed:\n" + "\n".join(f"- {issue}" for issue in issues))
    
    logging.info(f"Dataset structure validation passed: {data_path}")
    return True
