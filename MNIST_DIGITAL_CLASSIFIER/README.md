# MNIST Digital Classifier

A simple yet effective digit recognition system using PyTorch with two CNN architectures.

## Project Structure

```
MNIST_DIGITAL_CLASSIFIER/
├── README.md
├── requirements.txt
├── code/
│   ├── models/                      # Model definitions (just the architecture)
│   │   ├── __init__.py
│   │   ├── simple_cnn.py            # SimpleCNN class
│   │   └── better_cnn.py            # BetterCNN class
│   ├── train_simple_cnn.py          # Training script for SimpleCNN
│   ├── train_better_cnn.py          # Training script for BetterCNN
│   ├── predict.py                   # Prediction script (loads weights & predicts)
│   ├── visualize.py                 # Visualization script (shows feature maps)
│   └── weights/                     # Trained model weights (.pth files)
└── data/
    └── MNIST/                       # MNIST dataset
```

## Usage

### 1. Training

Train the SimpleCNN model:
```bash
cd code
python train_simple_cnn.py
```

Train the BetterCNN model:
```bash
cd code
python train_better_cnn.py
```

Both scripts will:
- Download MNIST dataset (if not already present)
- Train the model for 3 epochs
- Evaluate on test data
- Save weights to `weights/mnist_cnn.pth` or `weights/mnist_better_cnn.pth`

### 2. Prediction

Make predictions on a new image:
```bash
cd code
python predict.py
```

The script loads the trained BetterCNN weights and outputs the top 3 predictions with confidence scores.

### 3. Visualization

Visualize what the model "sees" in the first layer:
```bash
cd code
python visualize.py
```

Shows the 32 feature maps from the first convolution layer as a 4x8 grid.

## Models

### SimpleCNN
- Simple 2-layer CNN
- Fewer parameters, faster training
- Good for quick experiments

### BetterCNN
- Deeper architecture with more filters
- Data augmentation during training
- Dropout for regularization
- Better accuracy on test set

## Files Explained

- **models/*.py**: Only contain model class definitions (no training logic)
- **train_*.py**: Data loading + training loop + weight saving
- **predict.py**: Load weights + inference on new images
- **visualize.py**: Load weights + visualize internal representations
- **weights/**: Trained model parameters saved as PyTorch state dicts
