# this is because i had to flip the images for up down and left right of left hand so that i could use it for training instead of making a new dataset

import os
import cv2
os.chdir('dataset')


input_frame_dir = "down/frame"
input_raw_dir = "down/raw"
output_frame_dir = "down_for_right_hand/frame"
output_raw_dir = "down_for_right_hand/raw"


os.makedirs(output_frame_dir, exist_ok=True)
os.makedirs(output_raw_dir, exist_ok=True)

def flip_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        
        img = cv2.imread(input_path)
        if img is not None:
            flipped_img = cv2.flip(img, 1)
            cv2.imwrite(output_path, flipped_img)
            print(f"Flipped and saved: {output_path}")
        else:
            print(f"Could not read: {input_path}")

flip_images(input_frame_dir, output_frame_dir)
flip_images(input_raw_dir, output_raw_dir)

print("Flipping completed!")
