markdown
# Installation Guide

## Prerequisites
- Python 3.7 or higher
- pip package manager

## Installation Methods

### Method 1: Install from requirements.txt
```bash
git clone https://github.com/mahdi-nasiri-far/medical-image-segmentation-unet.git
cd medical-image-segmentation-unet
pip install -r requirements.txt
```
### Method 2: Install as package (Development)
```bash
pip install -e .
```
### Method 3: Using conda

```bash
conda create -n unet-env python=3.8
conda activate unet-env
pip install -r requirements.txt
```
### Verification
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

from src import UNet
print("U-Net model imported successfully!")
```
