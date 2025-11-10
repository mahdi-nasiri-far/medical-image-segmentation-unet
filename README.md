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
