# Usage Guide

## Basic Training

```python
from src.model import UNet
from src.data_generator import MedicalImageGenerator

# Initialize model
model = UNet(input_shape=(256, 256, 1))
model.compile(learning_rate=1e-4)

# Data augmentation
aug_params = {
    'rotation_range': 0.2,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.05,
    'zoom_range': 0.05,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Create generator
train_gen = MedicalImageGenerator(
    batch_size=4,
    train_path='data/train',
    image_folder='images',
    mask_folder='masks',
    augmentation_dict=aug_params
)

# Train model
history = model.model.fit(
    train_gen,
    steps_per_epoch=400,
    epochs=60
)
```
Prediction
```python
from src.data_generator import test_generator
import numpy as np

# Load test images
test_gen = test_generator("data/test")
predictions = model.model.predict(test_gen)

# Save results
for i, pred in enumerate(predictions):
    np.save(f"prediction_{i}.npy", pred)
```
Visualization
```python
from src.utils import plot_training_history

# Plot training history
plot_training_history(history)
text
```
