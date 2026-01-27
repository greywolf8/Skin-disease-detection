"""
Evaluation script for ISIC 2016 skin disease classification
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support
)
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

from model import load_model_checkpoint
from dataset import create_test_loader
from config import DEVICE, CLASS_NAMES


def evaluate_model(model_path, data_dir, metadata_file, batch_size=32, output_dir='results'):
    """Evaluate the trained model"""
    
    logging.info("="*50)
    logging.info("STARTING EVALUATION")
    logging.info("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model_name = checkpoint.get('model_name', 'resnet50')
        model, _, _ = load_model_checkpoint(model_path, model_name)
        model.eval()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise
    
    # Create test data loader
    logging.info("Creating test data loader...")
    test_loader, image_ids, true_labels = create_test_loader(
        data_dir=data_dir,
        metadata_file=metadata_file,
        batch_size=batch_size
    )
    
    # Evaluate model
    logging.info("Running evaluation...")
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    logging.info(f"Evaluated {len(y_true)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    
    # Detailed classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES,
        output_dict=True
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve (for binary classification)
    if len(CLASS_NAMES) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Class-wise performance
    class_performance = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            class_performance[class_name] = {
                'accuracy': class_acc,
                'precision': class_report[str(i)]['precision'] if str(i) in class_report else 0,
                'recall': class_report[str(i)]['recall'] if str(i) in class_report else 0,
                'f1-score': class_report[str(i)]['f1-score'] if str(i) in class_report else 0,
                'support': int(class_report[str(i)]['support']) if str(i) in class_report else 0
            }
    
    # Per-class accuracy bar plot
    if class_performance:
        classes = list(class_performance.keys())
        accuracies = [class_performance[cls]['accuracy'] for cls in classes]
        
        plt.figure(figsize=(10, 6))
        plt.bar(classes, accuracies, color=['lightblue', 'lightcoral'])
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Prediction confidence distribution
    confidence_scores = np.max(y_prob, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidence_scores):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'model_path': model_path,
        'total_samples': len(y_true),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        },
        'class_performance': class_performance,
        'confusion_matrix': cm.tolist(),
        'mean_confidence': float(np.mean(confidence_scores)),
        'std_confidence': float(np.std(confidence_scores))
    }
    
    if len(CLASS_NAMES) == 2:
        results['roc_auc'] = float(roc_auc)
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    predictions_df_data = {
        'image_id': image_ids,
        'true_label': y_true.tolist(),
        'predicted_label': y_pred.tolist(),
        'confidence': confidence_scores.tolist(),
        'prob_benign': y_prob[:, 0].tolist(),
        'prob_malignant': y_prob[:, 1].tolist() if y_prob.shape[1] > 1 else [0] * len(y_prob)
    }
    
    import pandas as pd
    predictions_df = pd.DataFrame(predictions_df_data)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Find misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    if len(misclassified_indices) > 0:
        misclassified_data = {
            'image_id': [image_ids[i] for i in misclassified_indices],
            'true_label': [CLASS_NAMES[y_true[i]] for i in misclassified_indices],
            'predicted_label': [CLASS_NAMES[y_pred[i]] for i in misclassified_indices],
            'confidence': [confidence_scores[i] for i in misclassified_indices]
        }
        
        misclassified_df = pd.DataFrame(misclassified_data)
        misclassified_path = os.path.join(output_dir, 'misclassified_samples.csv')
        misclassified_df.to_csv(misclassified_path, index=False)
        
        logging.info(f"Found {len(misclassified_indices)} misclassified samples")
        logging.info(f"Misclassified samples saved to {misclassified_path}")
    
    logging.info(f"Evaluation results saved to {results_path}")
    logging.info(f"Predictions saved to {predictions_path}")
    logging.info("Evaluation completed successfully!")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"{'='*50}")
    print(f"Total samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if len(CLASS_NAMES) == 2:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Mean Confidence: {np.mean(confidence_scores):.4f}")
    print(f"Misclassified: {len(misclassified_indices)}/{len(y_true)} ({100*len(misclassified_indices)/len(y_true):.2f}%)")
    print(f"Results saved to: {output_dir}")
    
    return results
