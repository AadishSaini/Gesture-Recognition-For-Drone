import os
import cv2
os.chdir('dataset')
# Directories
input_frame_dir = "down/frame"
input_raw_dir = "down/raw"
output_frame_dir = "down_for_right_hand/frame"
output_raw_dir = "down_for_right_hand/raw"

# Create output directories if they don't exist
os.makedirs(output_frame_dir, exist_ok=True)
os.makedirs(output_raw_dir, exist_ok=True)

def flip_images(input_dir, output_dir):
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read the image
        img = cv2.imread(input_path)
        if img is not None:
            # Flip the image horizontally
            flipped_img = cv2.flip(img, 1)
            # Save the flipped image to the output directory
            cv2.imwrite(output_path, flipped_img)
            print(f"Flipped and saved: {output_path}")
        else:
            print(f"Could not read: {input_path}")

# Flip images in both directories
flip_images(input_frame_dir, output_frame_dir)
flip_images(input_raw_dir, output_raw_dir)

print("Flipping completed!")
