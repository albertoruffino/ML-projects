# Physics-Informed Neural Networks (PINNs) Projects

This repository contains two projects that implement **Physics-Informed Neural Networks (PINNs)** to solve classical physics problems. Both projects use **PyTorch** and demonstrate how to encode physical laws directly into the training of neural networks.

---

## Physics Background

### Damped Harmonic Oscillator
The **damped harmonic oscillator** is a fundamental system in classical mechanics, describing the motion of a mass attached to a spring with friction (or damping). Its equation of motion is:


$m \ddot{u}(t) + \mu \dot{u}(t) + k u(t) = 0$


Where:  
- \( m \) = mass  
- \( k \) = spring constant  
- \( $\mu$ \) = damping coefficient  
- \( u(t) \) = displacement from equilibrium  

The system behavior depends on the damping ratio:  
- **Underdamped** $(\mu^2 < 4mk)$ → oscillatory decay  
- **Critically damped** $(\mu^2 = 4mk)$ → fastest return to equilibrium without oscillation  
- **Overdamped** $(\mu^2 > 4mk)$ → slow return to equilibrium without oscillation  

The analytical solution for the underdamped case is:


$u(t) = e^{-\delta t} A \cos(\omega t + \phi)$


with:  
$\delta = \frac{\mu}{2m}$ and $\omega = \sqrt{\frac{k}{m} - \delta^2}$

---

## Projects Overview

### 1. Classical PINN – Damped Harmonic Oscillator
- **Objective:** Solve the damped harmonic oscillator ODE using a PINN without labeled data.  
- **Method:**
  - Fully connected neural network with Tanh activations
  - Automatic differentiation for computing derivatives
  - Boundary conditions encoded directly in the network
  - Compare PINN predictions with exact analytical solution  
- **Notebook:** [`PINN_Damped_Harmonic_Oscillator.ipynb`](./PINN_Damped_Harmonic_Oscillator.ipynb)

### 2. PINN with Noisy Data
- **Objective:** Estimate the damping coefficient \( \mu \) using a PINN trained on **noisy observational data** and physics constraints.  
- **Method:**
  - Fully connected network with multiple hidden layers
  - Hybrid loss: combination of **data loss** (from noisy measurements) and **physics loss** (ODE residual)
  - Tracks the evolution of \(\mu\) during training
  - Visualizes PINN predictions vs exact solution  
- **Notebook:** [`PINN_Damped_Harmonic_Oscillator_NoisyData.ipynb`](./PINN_Damped_Harmonic_Oscillator_NoisyData.ipynb)

---

## Common Features
- Implemented in **Python** with **PyTorch**
- Leverages **automatic differentiation** for computing derivatives
- GPU support if available
- Detailed visualizations of training loss, predictions, and parameter estimation

---
