![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub stars](https://img.shields.io/github/stars/mahdi-nasiri-far/medical-image-segmentation-unet)
# Medical Image Segmentation with U-Net

![U-Net Architecture](https://github.com/mahdi-nasiri-far/medical-image-segmentation-unet/raw/main/docs/unet-architecture.png)

A complete implementation of U-Net architecture for biomedical image segmentation, based on the original paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

## 🎯 Features

- **Complete U-Net Architecture**: Encoder-decoder with skip connections
- **Medical Image Focus**: Optimized for biomedical image segmentation
- **Data Augmentation**: Real-time augmentation for medical imaging
- **Flexible Data Generator**: Custom data loader for image-mask pairs
- **Model Checkpoints**: Automatic saving of best performing models
- **Training Visualization**: Comprehensive loss and accuracy tracking
- **Easy Inference**: Simple API for prediction on new images

## 📊 Performance

After 60 epochs of training on membrane dataset:
- **Training Accuracy**: 97.91%
- **Training Loss**: 0.0640
- **Validation Accuracy**: 97.91% 
- **Validation Loss**: 0.0637

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mahdi-nasiri-far/medical-image-segmentation-unet.git
cd medical-image-segmentation-unet
pip install -r requirements.txt
```
### Basic Usage
```python
from src.model import UNet
from src.data_generator import MedicalImageGenerator

# Initialize model
model = UNet(input_shape=(256, 256, 1))
model.compile()

# Train model
history = model.model.fit(
    train_generator,
    steps_per_epoch=400,
    epochs=60
)
```
### Training with Custom Data
```python
from src.data_generator import MedicalImageGenerator

# Data augmentation for medical images
augmentation_params = {
    'rotation_range': 0.2,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.05,
    'zoom_range': 0.05,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Create data generator
train_gen = MedicalImageGenerator(
    batch_size=4,
    train_path='data/train',
    image_folder='images',
    mask_folder='masks',
    augmentation_dict=augmentation_params
)

# Train model
history = model.model.fit(
    train_gen,
    steps_per_epoch=400,
    epochs=60
)
```
### 🏗️ Model Architecture
```text
Input (256, 256, 1)
↓
Encoder (Contracting Path)
├── Conv 64 → Conv 64 → MaxPool
├── Conv 128 → Conv 128 → MaxPool  
├── Conv 256 → Conv 256 → MaxPool
├── Conv 512 → Conv 512 → Dropout → MaxPool
└── Conv 1024 → Conv 1024 → Dropout

Decoder (Expanding Path) + Skip Connections
├── UpConv 512 + Skip → Conv 512 → Conv 512
├── UpConv 256 + Skip → Conv 256 → Conv 256
├── UpConv 128 + Skip → Conv 128 → Conv 128
├── UpConv 64 + Skip → Conv 64 → Conv 64
└── Output Conv (1, sigmoid)
```
### 📁 Project Structure
```text
medical-image-segmentation-unet/
├── src/                 # Source code
│   ├── model.py        # U-Net architecture
│   ├── data_generator.py # Medical data loader
│   ├── train.py        # Training utilities
│   ├── predict.py      # Inference functions
│   └── utils.py        # Helper functions
├── notebooks/          # Jupyter notebooks
├── config/            # Configuration files
├── examples/          # Sample data and results
└── docs/              # Documentation
```
🛠️ Requirements
TensorFlow 2.x

Keras

OpenCV

scikit-image

NumPy

Matplotlib

scipy

pandas

📚 Documentation
Installation Guide

Usage Examples

🤝 Contributing
We welcome contributions! Please feel free to submit issues and pull requests.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Original U-Net paper by Olaf Ronneberger et al.

zhixuhao for the original implementation reference

Medical imaging research community

📧 Contact
Mahdi Nasiri Far - GitHub

Project Link: https://github.com/mahdi-nasiri-far/medical-image-segmentation-unet
