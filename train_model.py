import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



dataset_path = 'dataset/'


image_size = 128  
num_classes = len(os.listdir(dataset_path)) 

# train and test
X = []
y = []


for idx, gesture in enumerate(os.listdir(dataset_path)):
    gesture_path = os.path.join(dataset_path, gesture, 'frame')  
    if os.path.isdir(gesture_path):
        for img_file in os.listdir(gesture_path):
            # preprocess
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            img = cv2.resize(img, (image_size, image_size)) 
            X.append(img)
            y.append(idx) 

    # list of gesture name wrt the idx for a mistake i made earlier :pray:
    gesture_mapping = {idx: gesture for idx, gesture in enumerate(os.listdir(dataset_path))}


# numpy-ification go brr
X = np.array(X, dtype='float32') / 255.0  
X = np.expand_dims(X, axis=-1) 
y = np.array(y)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# model
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
    Dense(num_classes, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=30 # idk
)


#evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Save the model
model.save('final_model.h5')
print("hogaya")


print(gesture_mapping)
