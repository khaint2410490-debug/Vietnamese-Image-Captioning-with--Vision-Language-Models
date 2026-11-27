#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating package manager and system tools..."
pip install --upgrade pip

echo "Installing third-party external Python packages..."
# matplotlib, numpy, pandas, Pillow (for PIL)
pip install matplotlib numpy pandas Pillow

# Installing PyTorch and torchvision (CPU version by default; modify if CUDA is needed)
pip install torch torchvision

echo "--------------------------------------------------------"
echo "Setup Complete!"
echo "--------------------------------------------------------"
echo "Note: The following imports are local project files and"
echo "do not need to be downloaded via pip. Make sure they are"
echo "present in your project directory root:"
echo " - evaluation/ (denoise_metrics, evaluation, generate_caption)"
echo " - model/ (captioning, denoising)"
echo " - preprocessing/ (noise_generator, tokenizer)"
echo " - training/ (train)"
echo " - project_paths.py"
