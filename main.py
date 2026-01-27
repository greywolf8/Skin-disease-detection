#!/usr/bin/env python3
"""
ISIC 2016 Skin Disease Classification CLI
Binary classification of dermoscopic images (Benign vs Malignant)
"""

import argparse
import os
import sys
from pathlib import Path

from train import train_model
from evaluate import evaluate_model
from utils import predict_single_image, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="ISIC 2016 Skin Disease Binary Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --data-dir ISIC_2016 --epochs 50 --batch-size 32

  # Evaluate existing model
  python main.py evaluate --model-path checkpoints/best_model.pth --data-dir ISIC_2016

  # Predict single image
  python main.py predict --model-path checkpoints/best_model.pth --image path/to/image.jpg
  
  # Test model on test dataset  
  python main.py test-dataset --model-path checkpoints/best_model.pth --data-dir ISIC_2016
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-dir', type=str, required=True,
                             help='Path to ISIC_2016 dataset directory')
    train_parser.add_argument('--metadata-file', type=str,
                             default='metadata/ISIC_2016_Training_Metadata.csv',
                             help='Path to metadata CSV file')
    train_parser.add_argument('--model-name', type=str, default='resnet50',
                             choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3'],
                             help='Model architecture to use')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--val-split', type=float, default=0.2,
                             help='Validation split ratio')
    train_parser.add_argument('--output-dir', type=str, default='checkpoints',
                             help='Directory to save model checkpoints')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume training from')
    train_parser.add_argument('--weighted-loss', action='store_true',
                             help='Use weighted loss for class imbalance')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model checkpoint')
    eval_parser.add_argument('--data-dir', type=str, required=True,
                            help='Path to ISIC_2016 dataset directory')
    eval_parser.add_argument('--metadata-file', type=str,
                            default='metadata/ISIC_2016_Training_Metadata.csv',
                            help='Path to metadata CSV file')
    eval_parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for evaluation')
    eval_parser.add_argument('--output-dir', type=str, default='results',
                            help='Directory to save evaluation results')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Predict single image')
    predict_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to trained model checkpoint')
    predict_parser.add_argument('--image', type=str, required=True,
                               help='Path to image file')
    predict_parser.add_argument('--model-name', type=str, default='resnet50',
                               choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3'],
                               help='Model architecture used')
    
    # Test dataset command
    test_dataset_parser = subparsers.add_parser('test-dataset', help='Test model on test dataset')
    test_dataset_parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pth',
                                    help='Path to trained model checkpoint')
    test_dataset_parser.add_argument('--data-dir', type=str, default='ISIC_2016',
                                    help='Path to ISIC_2016 dataset directory')
    test_dataset_parser.add_argument('--batch-size', type=int, default=32,
                                    help='Batch size for testing')
    test_dataset_parser.add_argument('--model-name', type=str, default='resnet50',
                                    choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3'],
                                    help='Model architecture used')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'train':
            # Validate data directory
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"Error: Data directory '{data_dir}' does not exist")
                sys.exit(1)
            
            train_dir = data_dir / 'train'
            metadata_path = data_dir / args.metadata_file
            
            if not train_dir.exists():
                print(f"Error: Training directory '{train_dir}' does not exist")
                sys.exit(1)
            
            if not metadata_path.exists():
                print(f"Error: Metadata file '{metadata_path}' does not exist")
                sys.exit(1)
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Train model
            train_model(
                data_dir=str(data_dir),
                metadata_file=str(metadata_path),
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                val_split=args.val_split,
                output_dir=args.output_dir,
                resume_checkpoint=args.resume,
                use_weighted_loss=args.weighted_loss
            )
            
        elif args.command == 'evaluate':
            # Validate model path
            if not os.path.exists(args.model_path):
                print(f"Error: Model file '{args.model_path}' does not exist")
                sys.exit(1)
            
            # Validate data directory
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"Error: Data directory '{data_dir}' does not exist")
                sys.exit(1)
            
            metadata_path = data_dir / args.metadata_file
            if not metadata_path.exists():
                print(f"Error: Metadata file '{metadata_path}' does not exist")
                sys.exit(1)
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Evaluate model
            evaluate_model(
                model_path=args.model_path,
                data_dir=str(data_dir),
                metadata_file=str(metadata_path),
                batch_size=args.batch_size,
                output_dir=args.output_dir
            )
            
        elif args.command == 'predict':
            # Validate model path
            if not os.path.exists(args.model_path):
                print(f"Error: Model file '{args.model_path}' does not exist")
                sys.exit(1)
            
            # Validate image path
            if not os.path.exists(args.image):
                print(f"Error: Image file '{args.image}' does not exist")
                sys.exit(1)
            
            # Predict single image
            result = predict_single_image(
                model_path=args.model_path,
                image_path=args.image,
                model_name=args.model_name
            )
            
            print(f"\nPrediction Results:")
            print(f"Image: {args.image}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities:")
            print(f"  Benign: {result['prob_benign']:.4f}")
            print(f"  Malignant: {result['prob_malignant']:.4f}")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
