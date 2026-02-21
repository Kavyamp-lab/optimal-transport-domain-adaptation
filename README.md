# Optimal Transport for Domain Adaptation (MNIST)

This project demonstrates domain adaptation using Optimal Transport.

## Problem
A neural network trained on clean MNIST digits fails when tested on rotated and noisy digits (domain shift).

## Solution
We apply Sinkhorn Optimal Transport to align feature distributions between source and target domains.

## Results
- Wasserstein distance reduced after alignment
- Target accuracy improved after transport

## Methods Used
- CNN Feature Extractor (PyTorch)
- Sinkhorn Optimal Transport (POT library)
- Wasserstein Distance
- t-SNE visualization
- Logistic Regression baseline

## How to Run

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pot
python wasserstein_da.py