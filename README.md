# MNIST Neural Network Implementation

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification. The implementation demonstrates the evolution of the architecture through iterative improvements and optimization techniques.

## 🎯 Implementation Highlights

### Key Architecture Features
- **📊 Total Parameter Count**: 18,202 trainable parameters (lightweight and efficient)
- **⚡ Batch Normalization**: Applied after each convolutional layer for training stability
- **🛡️ Dropout Regularization**: Strategic 0.05 dropout before final classification layer
- **🌐 Global Average Pooling (GAP)**: Replaces traditional fully connected layers, reducing parameters while maintaining performance
- **🎯 Final Performance**: 99.49% test accuracy on MNIST dataset

### Training Configuration
- **Optimizer**: Adam (lr=0.001) with ReduceLROnPlateau scheduler
- **Batch Size**: 64 (optimized through iterative testing of 512, 128, 64, 32)
- **Data Augmentation**: Random rotation (±10°) and translation (20% of image size)
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Epochs**: 20 with adaptive learning rate reduction

### Architecture Evolution Journey
1. **Started without Batch Normalization** → Added for training stability
2. **Tested Dropout at each layer** → Removed due to performance impact, kept only before final layer
3. **Upgraded from SGD+StepLR to Adam+ReduceLROnPlateau** → Improved convergence and final accuracy
4. **Batch Size Optimization** → Tested 512, 128, 64, 32 → Selected 64 for optimal performance/memory balance

## Dataset

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Task**: Handwritten digit classification (0-9)
- **Input Size**: 28×28 grayscale images
- **Classes**: 10 digits (0-9)
- **Training Samples**: 60,000
- **Test Samples**: 10,000

## Data Preprocessing

### Training Transformations
```python
transforms.Compose([
    transforms.RandomRotation(10, fill=(0,)),           # 🔄 Random rotations (±10°)
    transforms.RandomAffine(0, translate=(0.2, 0.2), fill=0),  # 📍 Random translations (20% of image)
    transforms.ToTensor(),                              # Convert to tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))         # MNIST normalization
])
```

### Data Augmentation Benefits
- **🔄 Random Rotation (±10°)**: Improves model robustness to slight digit orientation variations
- **📍 Random Translation (20% of image size)**: Enhances generalization to different digit positions
- **Combined Effect**: Significantly improves model's ability to handle real-world variations in handwritten digits

### Test Transformations
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## Neural Network Architecture

### Model: SimpleMNIST

The final architecture consists of 5 convolutional layers followed by Global Average Pooling and a fully connected classifier:

```
Input: [Batch, 1, 28, 28]
    ↓
Conv2d(1→8, 3×3, padding=1) + BatchNorm2d(8) + ReLU
    ↓ MaxPool2d(2×2)
Conv2d(8→16, 3×3, padding=1) + BatchNorm2d(16) + ReLU  
    ↓ MaxPool2d(2×2)
Conv2d(16→32, 3×3, padding=1) + BatchNorm2d(32) + ReLU
    ↓
Conv2d(32→32, 3×3, padding=1) + BatchNorm2d(32) + ReLU
    ↓
Conv2d(32→64, 1×1) + BatchNorm2d(64) + ReLU
    ↓ Global Average Pooling (1×1)
Dropout(0.1)
    ↓
Linear(64→10)
    ↓
Output: [Batch, 10]
```

### Architecture Details

- **Total Parameters**: 18,202 trainable parameters
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Mode: 'max' (monitoring accuracy)
  - Factor: 0.2 (reduce LR by 80%)
  - Patience: 2 epochs
  - Threshold: 1e-4
  - Min LR: 1e-5
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Epochs**: 20

### Key Architecture Features

#### ✅ Batch Normalization
- Applied after each convolutional layer
- Provides training stability and faster convergence
- Reduces internal covariate shift

#### ✅ Dropout
- Applied only before the final fully connected layer (0.05 dropout rate)
- Prevents overfitting while maintaining performance
- Initially tested at each layer but removed due to performance impact

#### ✅ Global Average Pooling (GAP)
- Replaces traditional fully connected layers
- Reduces parameter count significantly
- Provides spatial feature aggregation
- Acts as a regularizer

#### ✅ Fully Connected Layer
- Final classification layer: Linear(64→10)
- Maps GAP output to 10 class probabilities

## Training Progress and Architecture Evolution

### 1. Initial Architecture (Without Batch Normalization)
- **Problem**: Training instability and slower convergence
- **Solution**: Added Batch Normalization after each convolutional layer
- **Result**: Significant improvement in training stability and convergence speed

### 2. Dropout Implementation and Optimization
- **Initial Approach**: Applied dropout after each layer
- **Problem**: Performance degradation observed
- **Solution**: Removed dropout from intermediate layers, kept only before final FC layer
- **Result**: Better performance with reduced overfitting

### 3. Optimizer and Scheduler Improvements
- **Initial**: SGD with StepLR scheduler
  - Learning Rate: 0.05
  - Momentum: 0.9
  - Weight Decay: 1e-4
  - Step Size: 15 epochs, Gamma: 0.1
- **Final**: Adam with ReduceLROnPlateau
  - Learning Rate: 0.001
  - Weight Decay: 1e-4
  - Adaptive learning rate reduction based on validation accuracy
- **Result**: Improved convergence and final accuracy

### 4. Batch Size Optimization
- **Tested Values**: 512, 128, 64, 32
- **Selection Criteria**: Balance between training stability, memory efficiency, and performance
- **Final Choice**: 64
  - Provides good gradient estimates
  - Efficient memory usage
  - Optimal convergence speed
  - Best performance on MNIST dataset

## Performance Results

The model achieves excellent performance on the MNIST dataset:

- **Final Test Accuracy**: 99.49% (9,949/10,000 correct predictions)
- **Training Stability**: Consistent improvement across epochs
- **Convergence**: Stable training with minimal overfitting

## Key Implementation Features

### Data Augmentation
- **🔄 Random Rotation**: ±10 degrees to handle digit orientation variations
- **📍 Random Translation**: 20% of image size for position invariance
- **Impact**: Significantly improves generalization and robustness to real-world digit variations

### Training Configuration
- **Device**: CUDA-enabled GPU training
- **Data Loading**: 2 workers with pin memory for efficiency
- **Progress Tracking**: Real-time accuracy and loss monitoring

### Loss Function Enhancement
- **Label Smoothing**: 0.1 smoothing factor
- **Purpose**: Prevents overconfident predictions and improves generalization

## File Structure

```
soai_session5/
├── soai_session5.ipynb    # Main implementation notebook
└── README.md              # This documentation
```

## Usage

1. Ensure PyTorch and required dependencies are installed
2. Run the notebook cells sequentially
3. The model will automatically download MNIST dataset
4. Training will commence with real-time progress tracking
5. Final model performance will be displayed

## Dependencies

- PyTorch
- Torchvision
- Matplotlib
- tqdm
- collections

## Key Takeaways

1. **Batch Normalization** is crucial for training stability in deep CNNs
2. **Strategic Dropout Placement** is more important than applying it everywhere
3. **Adam Optimizer** with adaptive learning rate scheduling outperforms SGD for this task
4. **Global Average Pooling** significantly reduces parameters while maintaining performance
5. **Data Augmentation** helps improve model generalization
6. **Label Smoothing** provides additional regularization benefits

This implementation demonstrates the iterative process of neural network optimization, showing how each architectural and training modification contributes to the final high-performance model.
