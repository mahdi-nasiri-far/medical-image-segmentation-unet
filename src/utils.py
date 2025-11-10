"""
Utility functions for U-Net medical image segmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def plot_training_history(history):
    """
    Plot training history including accuracy and loss
    
    Args:
        history: Training history object from model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], 'g--', label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], 'r-', label='Train Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], 'y--', label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def get_model_checkpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True):
    """
    Create model checkpoint callback
    
    Args:
        filepath: Path to save model
        monitor: Metric to monitor
        verbose: Verbosity mode
        save_best_only: Whether to save only the best model
    
    Returns:
        ModelCheckpoint callback
    """
    return ModelCheckpoint(
        filepath,
        monitor=monitor,
        verbose=verbose,
        save_best_only=save_best_only,
        save_weights_only=False
    )


def save_result(save_path, results, threshold=0.5):
    """
    Save segmentation results as images
    
    Args:
        save_path: Directory to save results
        results: Prediction results from model
        threshold: Threshold for binary segmentation
    """
    import os
    import skimage.io as io
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i, item in enumerate(results):
        img = (item[:, :, 0] * 255).astype(np.uint8)
        
        # Apply threshold for binary segmentation
        if threshold is not None:
            img = (img > threshold * 255).astype(np.uint8) * 255
            
        io.imsave(os.path.join(save_path, f"pred_{i}.png"), img)


def calculate_iou(y_true, y_pred, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) metric
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        threshold: Threshold for binary segmentation
    
    Returns:
        IoU score
    """
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > threshold).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    
    return intersection / (union + 1e-7)


def visualize_prediction(original, ground_truth, prediction, threshold=0.5):
    """
    Visualize original image, ground truth and prediction
    
    Args:
        original: Original input image
        ground_truth: Ground truth mask
        prediction: Model prediction
        threshold: Threshold for binary segmentation
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original.squeeze(), cmap='gray')
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction.squeeze(), cmap='gray')
    axes[2].set_title('Prediction', fontweight='bold')
    axes[2].axis('off')
    
    # Binary prediction
    binary_pred = (prediction > threshold).astype(np.float32)
    axes[3].imshow(binary_pred.squeeze(), cmap='gray')
    axes[3].set_title(f'Binary (threshold={threshold})', fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
