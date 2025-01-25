# Gesture Recognition For Drone


### files in the project

datagen.py - creates the dataset
datagen_for_right.py - for flipping the left hand images to convert to right hand
train_model.py - the model training 
test.py - main python file
final_model.h5 - the final model compiled


## Dataset Samples

Below are some sample images from the dataset used for training:

### Backward Gesture
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png)

### Forward Gesture
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/forward/frame/frame_1.png)

### Up Gesture
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/up/frame/frame_1.png)

### Down Gesture
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/down/frame/frame_1.png)

### Left Gesture
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/left/frame/frame_1.png)

### Right Gesture
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/right/frame/frame_1.png)

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python test.py
   ```

## Model Overview

The model was trained using a dataset of hand gestures for left and right hands, including gestures  "up", "down", "left", "right", "forward" and "backward".&#x20;

### Model Architecture:

1. **Conv2D Layer 1**: Applies 32 filters of size 3x3 with ReLU activation to extract features from the input image.
2. **MaxPooling2D Layer 1**: Reduces the spatial dimensions of the feature map using a 2x2 pooling operation.
3. **Dropout 1**: Regularization to prevent overfitting, dropping 25% of the neurons.
4. **Conv2D Layer 2**: Applies 64 filters of size 3x3 with ReLU activation to extract more complex features.
5. **MaxPooling2D Layer 2**: Further reduces the spatial dimensions using a 2x2 pooling operation.
6. **Dropout 2**: Regularization, dropping 25% of the neurons.
7. **Flatten Layer**: Flattens the 2D feature maps into a 1D vector to feed into the fully connected layers.
8. **Dense Layer 1**: Fully connected layer with 128 units and ReLU activation to learn complex patterns.
9. **Dropout 3**: Regularization, dropping 50% of the neurons.
10. **Dense Layer 2 (Output Layer)**: A fully connected layer with `num_classes` units and softmax activation for multi-class classification.

Preprocessing steps include resizing, grayscale conversion, and normalization.

## Dataset Structure

The dataset is organized as follows:

```
dataset/
  left/
    frame/
    raw/
  right/
    frame/
    raw/
```

where raw is the images with the mediapipe frame over a black background (cause i thought might be useful in training) and frame is the cropped image of hand.

### Working
![alt text](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/Screenshot%202025-01-25%20205217.png)
