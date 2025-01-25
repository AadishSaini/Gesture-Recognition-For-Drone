# Gesture Recognition For Drone

## Dataset Samples

Below are some sample images from the dataset used for training:

### Backward Gesture

![alt text]\([https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame\_1.png](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png))

### Upward Gesture

![alt text]\([https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/upward/frame/frame\_1.png](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png))

### Up Gesture

![alt text]\([https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/up/frame/frame\_1.png](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png))

### Down Gesture

![alt text]\([https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/down/frame/frame\_1.png](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png))

### Left Gesture

![alt text]\([https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/left/frame/frame\_1.png](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png))

### Right Gesture

![alt text]\([https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/right/frame/frame\_1.png](https://github.com/AadishSaini/Gesture-Recognition-For-Drone/blob/main/dataset/backward/frame/frame_1.png))

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

