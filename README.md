# From-Scratch Implementation of LeNet-5 on MNIST  
*A Classical Convolutional Neural Network Revisited with Modern Tooling*

---

## Overview
This repository presents a **from-scratch implementation of the LeNet-5 convolutional neural network**, one of the earliest and most influential deep learning architectures, originally proposed for handwritten digit recognition. The implementation is intentionally minimalistic and faithful to the original design philosophy, emphasizing **architectural understanding, historical grounding, and experimental clarity**.

The project serves as a **foundational deep learning artifact**, appropriate for inclusion in a **PhD research portfolio**, particularly as evidence of conceptual mastery over canonical neural architectures.

---

## Research Motivation
LeNet-5 laid the groundwork for modern convolutional neural networks by introducing:
- Local receptive fields  
- Weight sharing via convolution  
- Subsampling through pooling  
- Hierarchical feature learning  

Re-implementing LeNet-5 from first principles provides:
- Architectural literacy in CNN design
- Insight into early activation functions and pooling strategies
- A baseline for comparing modern deep architectures against classical models

This work focuses on **understanding and reproducibility**, not architectural novelty.

---

## Dataset
The MNIST dataset is used as a standardized benchmark for handwritten digit classification.

### Dataset Characteristics
- Grayscale images of size 28×28
- 10 digit classes (0–9)
- Well-suited for validating classical CNN architectures

The dataset enables controlled experimentation without confounding complexity.

---

## Model Architecture: LeNet-5

The network follows the classical LeNet-5 pipeline:

1. **Convolution Layer**
   - 6 filters, 5×5 kernel
   - `tanh` activation
   - Same padding

2. **Average Pooling**
   - Subsampling for spatial reduction

3. **Convolution Layer**
   - 16 filters, 5×5 kernel
   - `tanh` activation
   - Valid padding

4. **Average Pooling**
   - Further spatial compression

5. **Fully Connected Layers**
   - Dense layer with 120 units (`tanh`)
   - Dense layer with 84 units (`tanh`)

6. **Output Layer**
   - 10-unit softmax classifier

This structure mirrors the original LeNet-5 design while leveraging modern deep learning frameworks for implementation clarity.

---

## Training Configuration
- Optimizer: Adam
- Loss Function: Sparse Categorical Cross-Entropy
- Metrics: Accuracy
- Validation Split: 20%
- Early Stopping:
  - Monitors training accuracy
  - Prevents overfitting
  - Ensures stable convergence

Training is intentionally limited in epochs to emphasize architectural validation rather than aggressive optimization.

---

## Evaluation and Analysis
The trained model is evaluated using:
- Test set accuracy
- Loss evaluation on unseen data
- Class probability inspection
- Argmax-based prediction decoding

Training dynamics are visualized through loss and accuracy curves to assess convergence behavior.

---

## Key Contributions
- Faithful reimplementation of LeNet-5
- Use of original activation functions (`tanh`, average pooling)
- Clear architectural transparency
- Early stopping–based training control
- Reproducible experimental setup

---

## Technologies Used
Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib

---

## Research Significance
This repository demonstrates **fundamental competency in convolutional neural networks**, which is essential for advanced research in:
- Computer vision
- Representation learning
- Deep learning theory

By revisiting LeNet-5 from scratch, this work establishes a strong conceptual baseline for extending toward modern architectures such as AlexNet, VGG, Inception, and beyond.

---

## Future Extensions
Potential extensions include:
- Comparative analysis with modern CNNs
- Activation function ablation studies
- Pooling strategy comparisons
- Parameter efficiency analysis
- Educational benchmarking for deep learning curricula

---

## Reproducibility Statement
All architectural choices, training configurations, and evaluation procedures are explicitly defined to ensure full reproducibility and methodological transparency.

---
