# ISIC 2016 Skin Disease Classification

A deep learning CLI application for binary classification of dermoscopic images to detect benign vs malignant skin lesions using the ISIC 2016 dataset.

## Features

- **Multiple Model Architectures**: ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3
- **Transfer Learning**: Pre-trained models fine-tuned for skin disease classification
- **Data Augmentation**: Comprehensive augmentation pipeline for better generalization
- **Class Imbalance Handling**: Weighted loss functions and stratified sampling
- **Comprehensive Evaluation**: ROC curves, confusion matrices, per-class metrics
- **Early Stopping**: Prevents overfitting with configurable patience
- **CLI Interface**: Easy-to-use command-line interface

## Dataset Structure

Your ISIC 2016 dataset should be organized as follows:

```
ISIC_2016/
├── train/
│   ├── ISIC_0000001.jpg
│   ├── ISIC_0000002.jpg
│   └── ...
├── test/ (optional)
│   ├── ISIC_0000901.jpg
│   └── ...
└── metadata/
    └── ISIC_2016_Training_Metadata.csv
```

## Installation

All required dependencies are already installed:
- PyTorch (CPU version)
- torchvision
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- PIL (Pillow)
- tqdm
- efficientnet-pytorch

## Usage

### 1. Training a Model

```bash
# Basic training with ResNet50
python main.py train --data-dir ISIC_2016 --epochs 50 --batch-size 32

# Training with EfficientNet and weighted loss for imbalanced data
python main.py train --data-dir ISIC_2016 --model-name efficientnet-b0 --epochs 30 --weighted-loss

# Training with custom parameters
python main.py train \
    --data-dir ISIC_2016 \
    --model-name resnet101 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --val-split 0.25 \
    --output-dir my_checkpoints
```

### 2. Evaluating a Model

```bash
# Basic evaluation
python main.py evaluate --model-path checkpoints/best_model.pth --data-dir ISIC_2016

# Evaluation with custom output directory
python main.py evaluate \
    --model-path checkpoints/best_model.pth \
    --data-dir ISIC_2016 \
    --output-dir evaluation_results
```

### 3. Predicting Single Images

```bash
# Predict a single image
python main.py predict \
    --model-path checkpoints/best_model.pth \
    --image ISIC_2016/train/ISIC_0000001.jpg

# Predict with specific model architecture
python main.py predict \
    --model-path checkpoints/best_model.pth \
    --image path/to/your/image.jpg \
    --model-name efficientnet-b0
```

## Model Architectures

- **ResNet50**: Good balance of performance and speed
- **ResNet101**: Higher capacity, better for complex patterns
- **EfficientNet-B0**: Efficient and accurate, good for resource constraints
- **EfficientNet-B3**: Higher accuracy, more parameters

## Training Features

- **Automatic validation split**: Stratified sampling maintains class balance
- **Data augmentation**: Random rotations, flips, color jittering
- **Learning rate scheduling**: Reduces LR on plateau
- **Early stopping**: Prevents overfitting
- **Checkpointing**: Regular model saves during training
- **Class weights**: Handles imbalanced datasets

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC curve and AUC
- Confusion matrix
- Per-class performance
- Confidence score distribution
- Misclassified samples analysis

## Output Files

### Training
- `checkpoints/best_model.pth`: Best performing model
- `checkpoints/final_model.pth`: Final model after training
- `checkpoints/training_history.png`: Loss and accuracy plots
- `checkpoints/training_summary.json`: Training statistics

### Evaluation
- `results/evaluation_results.json`: Detailed metrics
- `results/predictions.csv`: All predictions with probabilities
- `results/confusion_matrix.png`: Confusion matrix visualization
- `results/roc_curve.png`: ROC curve (binary classification)
- `results/misclassified_samples.csv`: Analysis of errors

## Example Workflow

1. **Prepare your data** in the required structure
2. **Train a model**:
   ```bash
   python main.py train --data-dir ISIC_2016 --model-name efficientnet-b0 --epochs 50 --weighted-loss
   ```
3. **Evaluate the model**:
   ```bash
   python main.py evaluate --model-path checkpoints/best_model.pth --data-dir ISIC_2016
   ```
4. **Make predictions**:
   ```bash
   python main.py predict --model-path checkpoints/best_model.pth --image new_lesion.jpg
   ```

## Tips for Best Results

1. **Use weighted loss** for imbalanced datasets: `--weighted-loss`
2. **Start with EfficientNet-B0** for good performance
3. **Monitor validation accuracy** to prevent overfitting
4. **Use larger batch sizes** if you have sufficient memory
5. **Experiment with learning rates** (0.001, 0.0001, 0.00001)

## Troubleshooting

- **Out of memory**: Reduce batch size (`--batch-size 16` or `--batch-size 8`)
- **Slow training**: Use smaller models (ResNet50 vs ResNet101)
- **Poor accuracy**: Try weighted loss, more epochs, or different models
- **Missing images**: Check dataset structure and file extensions

## Technical Details

- **Input size**: 224x224 RGB images
- **Normalization**: ImageNet statistics
- **Device**: Automatically detects CUDA or uses CPU
- **Precision**: Mixed precision training supported
- **Reproducibility**: Fixed random seeds for consistent results