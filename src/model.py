"""
U-Net Model Implementation for Medical Image Segmentation
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, 
    UpSampling2D, concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


class UNet:
    """
    U-Net Architecture for Medical Image Segmentation
    
    Based on: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Original Paper: https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self, input_shape=(256, 256, 1), num_classes=1, dropout_rate=0.5):
        """
        Initialize U-Net model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build U-Net architecture"""
        
        inputs = Input(self.input_shape)
        
        # Encoder (Contracting Path)
        # Block 1
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(self.dropout_rate)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bridge
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(self.dropout_rate)(conv5)
        
        # Decoder (Expanding Path) with Skip Connections
        # Block 6
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        
        # Block 7
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        
        # Block 8
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        
        # Block 9
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        
        # Output layer
        if self.num_classes == 1:
            outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
        else:
            outputs = Conv2D(self.num_classes, 1, activation='softmax')(conv9)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile(self, learning_rate=1e-4):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['accuracy', self.dice_coefficient]
        )
        
        return self.model
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1.0):
        """
        Dice coefficient metric for segmentation evaluation
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


# Backward compatibility
def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    """
    Legacy function for backward compatibility
    
    Args:
        pretrained_weights: Path to pre-trained weights
        input_size: Input image size
        
    Returns:
        Compiled U-Net model
    """
    model_builder = UNet(input_shape=input_size)
    model = model_builder.compile()
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model
