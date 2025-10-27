
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the MNIST dataset.  
It demonstrates **data preprocessing, data augmentation, PyTorch Dataset & DataLoader usage, CNN modeling, training, and evaluation**.

---

## Dataset
- **MNIST** dataset: 70,000 grayscale images of handwritten digits (28x28 pixels).  
- **Train/Test split**: Subsets of 2,000 training images and 500 test images are used for faster experimentation.  
- **Target labels**: Digits 0–9.  
- Source: [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/datasets.html#mnist)

---

## Technologies
- PyTorch  
- torchvision  
- Matplotlib  
- NumPy  

---

## Features
- **Data normalization** for better convergence  
- **Data augmentation**:
  - Random rotation  
  - Random horizontal flip  
  - Random affine transformations  
- CNN with:
  - 2 convolutional layers  
  - MaxPooling  
  - Fully connected layers  
  - Dropout for regularization  
- **Negative Log-Likelihood Loss (NLLLoss)** for multi-class classification  
- **Stochastic Gradient Descent (SGD)** optimizer  
- Accuracy tracking on train and test sets  
- Visualizations:
  - Training vs Test Accuracy per epoch  
  - Sample predictions with true vs predicted labels  

---

## Goal
Improve model generalization by applying **data augmentation** and leveraging a convolutional network architecture, achieving high accuracy on handwritten digit classification.

---