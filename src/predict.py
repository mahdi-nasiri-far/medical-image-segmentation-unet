"""
Prediction and inference utilities for U-Net model
"""

import numpy as np
import os
import tensorflow as tf
from .data_generator import test_generator
from .utils import save_result, visualize_prediction, calculate_iou


def predict(model, test_path, num_images=30, target_size=(256, 256)):
    """
    Generate predictions on test images
    
    Args:
        model: Trained U-Net model
        test_path: Path to test images
        num_images: Number of test images
        target_size: Target image size
    
    Returns:
        Array of predictions
    """
    test_gen = test_generator(test_path, num_images, target_size)
    predictions = model.predict(test_gen, verbose=1)
    return predictions


def predict_single_image(model, image_path, target_size=(256, 256)):
    """
    Predict segmentation for a single image
    
    Args:
        model: Trained U-Net model
        image_path: Path to single image
        target_size: Target image size
    
    Returns:
        Prediction array
    """
    import skimage.io as io
    import skimage.transform as trans
    
    # Load and preprocess image
    img = io.imread(image_path, as_gray=True)
    img = img / 255.0
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)
    
    # Predict
    prediction = model.predict(img, verbose=0)
    return prediction[0]


def batch_predict(model, image_dir, output_dir, target_size=(256, 256)):
    """
    Batch prediction on multiple images
    
    Args:
        model: Trained U-Net model
        image_dir: Directory containing input images
        output_dir: Directory to save predictions
        target_size: Target image size
    """
    import glob
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.tif"))
    
    print(f"Found {len(image_files)} images for prediction")
    
    for i, image_file in enumerate(image_files):
        # Predict
        prediction = predict_single_image(model, image_file, target_size)
        
        # Save prediction
        output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_file)}")
        save_single_prediction(prediction, output_path)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")


def save_single_prediction(prediction, output_path, threshold=0.5):
    """
    Save single prediction as image
    
    Args:
        prediction: Prediction array
        output_path: Output file path
        threshold: Threshold for binary segmentation
    """
    import skimage.io as io
    
    # Convert to uint8 image
    img = (prediction[:, :, 0] * 255).astype(np.uint8)
    
    # Apply threshold if specified
    if threshold is not None:
        img = (img > threshold * 255).astype(np.uint8) * 255
    
    # Save image
    io.imsave(output_path, img)


def evaluate_model_performance(model, test_images, test_masks, threshold=0.5):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained U-Net model
        test_images: Test images array
        test_masks: Test masks array
        threshold: Threshold for binary segmentation
    
    Returns:
        Dictionary with performance metrics
    """
    # Predict
    predictions = model.predict(test_images, verbose=1)
    
    # Calculate metrics
    iou_scores = []
    dice_scores = []
    
    for i in range(len(test_images)):
        iou = calculate_iou(test_masks[i], predictions[i], threshold)
        dice = 2 * iou / (1 + iou)  # Convert IoU to Dice
        iou_scores.append(iou)
        dice_scores.append(dice)
    
    metrics = {
        'mean_iou': np.mean(iou_scores),
        'std_iou': np.std(iou_scores),
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'min_iou': np.min(iou_scores),
        'max_iou': np.max(iou_scores)
    }
    
    return metrics, predictions


def load_model(model_path):
    """
    Load trained model from file
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    return tf.keras.models.load_model(model_path)


def create_prediction_pipeline(model_path, target_size=(256, 256)):
    """
    Create a complete prediction pipeline
    
    Args:
        model_path: Path to trained model
        target_size: Target image size
    
    Returns:
        Prediction function
    """
    model = load_model(model_path)
    
    def predict_pipeline(image_path, threshold=0.5):
        """
        Pipeline for predicting on single image
        
        Args:
            image_path: Path to input image
            threshold: Segmentation threshold
        
        Returns:
            Binary segmentation mask
        """
        prediction = predict_single_image(model, image_path, target_size)
        binary_mask = (prediction > threshold).astype(np.float32)
        return binary_mask
    
    return predict_pipeline
