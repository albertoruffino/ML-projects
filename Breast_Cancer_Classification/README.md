# Breast Cancer Classification with PyTorch

A neural network that predicts breast cancer (malignant or benign) from the scikit-learn dataset.  
This project demonstrates data preprocessing, normalization, PyTorch Dataset & DataLoader usage, model training, and performance evaluation.

The model is trained using a **fully connected neural network** with one hidden layer, and outputs probabilities using a sigmoid activation function.

---

## Dataset
The dataset contains 569 samples with 30 features each.  
Target labels: 0 = Malignant, 1 = Benign.  
Source: [scikit-learn breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

## Technologies
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

---

## Features
- Data normalization for better convergence  
- Train/test split (80% train, 20% test)  
- Fully connected neural network with one hidden layer  
- Binary Cross-Entropy Loss function (BCE)  
- Stochastic Gradient Descent optimizer (SGD)  
- Accuracy tracking on train and test sets  
- Visualizations:
  - Training vs Test Accuracy per epoch  
  - Training Loss per epoch  
  - Confusion Matrix on the test set  

---
