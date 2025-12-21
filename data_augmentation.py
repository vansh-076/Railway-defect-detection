import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths
data_dir = r"C:\Users\kvans\OneDrive\Desktop\class 3\non-defective"
output_dir = data_dir  

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Calculate number of augmentations needed
num_original_images = len(os.listdir(data_dir))
num_augmentations_per_image = (5000 - num_original_images) // num_original_images + 1

# Loop through original images and generate augmentations
for filename in tqdm(os.listdir(data_dir)):
    if filename.endswith(('.jpg', '.jpeg', '.webp')):  # Adjust file extensions if needed
        img_path = os.path.join(data_dir, filename)
        img = keras.preprocessing.image.load_img(img_path)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for ImageDataGenerator

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir,
                                  save_prefix='aug_', save_format='jpg'):
            i += 1
            if i >= num_augmentations_per_image:
                break