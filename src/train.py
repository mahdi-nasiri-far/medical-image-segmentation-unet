"""
Training utilities for U-Net model
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, CSVLogger
)
from .utils import plot_training_history


def train_model(model, train_generator, validation_generator=None,
                epochs=100, steps_per_epoch=1000, validation_steps=None,
                model_save_path='best_model.h5', patience=20):
    """
    Train U-Net model with callbacks
    
    Args:
        model: Compiled U-Net model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        validation_steps: Validation steps
        model_save_path: Path to save best model
        patience: Early stopping patience
    
    Returns:
        Training history
    """
    
    # Create callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss' if validation_generator else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss' if validation_generator else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss' if validation_generator else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Log training history
        CSVLogger('training_log.csv')
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_generator, steps=100):
    """
    Evaluate model on test data
    
    Args:
        model: Trained U-Net model
        test_generator: Test data generator
        steps: Number of test steps
    
    Returns:
        Evaluation metrics
    """
    evaluation = model.evaluate(test_generator, steps=steps, verbose=1)
    return dict(zip(model.metrics_names, evaluation))
