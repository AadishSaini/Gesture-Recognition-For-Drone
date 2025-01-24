import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# Set dataset path
dataset_path = 'dataset/'

# Hyperparameters
image_size = 128  # Resize all images to 128x128
num_classes = len(os.listdir(dataset_path))  # Number of gestures

# Data and labels
X = []
y = []

# Iterate through each gesture folder
for idx, gesture in enumerate(os.listdir(dataset_path)):
    gesture_path = os.path.join(dataset_path, gesture, 'frame')  # Use 'frame' images
    if os.path.isdir(gesture_path):
        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            img = cv2.resize(img, (image_size, image_size))  # Resize image
            X.append(img)
            y.append(idx)  # Assign a numerical label to each gesture
    # Create a mapping of gesture IDs to gesture names
    gesture_mapping = {idx: gesture for idx, gesture in enumerate(os.listdir(dataset_path))}


# Convert to NumPy arrays
X = np.array(X, dtype='float32') / 255.0  # Normalize pixel values to [0, 1]
X = np.expand_dims(X, axis=-1)  # Add channel dimension (for grayscale images)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=30
)


# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Save the model
model.save('final_model.h5')
print("Model saved as gesture_recognition_model.h5")


print(gesture_mapping)
