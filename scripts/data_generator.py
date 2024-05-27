#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[37]:


import os
import argparse
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from scipy.ndimage import label
from skimage.morphology import skeletonize, dilation, square
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def skeletonize_and_buffer(mask, buffer_size=3):
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    skeleton = skeletonize(mask)
    buffered_mask = dilation(skeleton, square(buffer_size))
    if mask.ndim == 3:
        buffered_mask = np.expand_dims(buffered_mask, axis=-1)
    return buffered_mask

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def remove_small_objects(mask, min_size=200):
    labeled_mask, num_features = label(mask)
    small_objects_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = np.where(labeled_mask == i, 1, 0)
        if np.sum(component) < min_size:
            small_objects_mask += component
    return mask - small_objects_mask

class DataGenerator(Sequence):
    def __init__(self, image_list, mask_list, batch_size=32, image_size=(256, 256), shuffle=True, augment=False, min_area=500, buffer_size=3, threshold=30):
        self.image_list = image_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.min_area = min_area
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.indices = np.arange(len(self.image_list))
        self.image_list, self.mask_list = self.validate_image_dimensions()
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.image_list) // self.batch_size

    def validate_image_dimensions(self):
        valid_image_list = []
        valid_mask_list = []
        required_size = self.image_size[0]  # Assuming image_size is a square (width, height)
        for img_path, mask_path in zip(self.image_list, self.mask_list):
            with rasterio.open(img_path) as src:
                if src.shape[0] < required_size or src.shape[1] < required_size:
                    continue
            valid_image_list.append(img_path)
            valid_mask_list.append(mask_path)
        print(f"Validated {len(valid_image_list)} images and {len(valid_mask_list)} masks.")
        return valid_image_list, valid_mask_list

    def augment_image(self, image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, mask

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_list[i] for i in indices]
        batch_mask_paths = [self.mask_list[i] for i in indices]
        
        X = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        valid_samples = 0
        
        for i, (img_path, mask_path) in enumerate(zip(batch_image_paths, batch_mask_paths)):
            with rasterio.open(img_path) as src:
                img = src.read(1)
                img = np.expand_dims(img, axis=-1)
                img = tf.image.resize(img, self.image_size)
                img = normalize_image(img)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                mask = np.expand_dims(mask, axis=-1)
                mask = tf.image.resize(mask, self.image_size)
                mask = tf.where(mask > self.threshold, 1, 0)
                mask = remove_small_objects(mask, min_size=self.min_area)
                mask = skeletonize_and_buffer(mask, buffer_size=self.buffer_size)
                if self.augment:
                    img, mask = self.augment_image(img, mask)
                img = img.numpy()
#                 mask = mask.numpy()
                if np.sum(mask) < self.min_area:
                    continue
                X[valid_samples] = img
                y[valid_samples] = np.expand_dims(mask, axis=-1)
                valid_samples += 1
                if valid_samples >= self.batch_size:
                    break
        
        while valid_samples < self.batch_size:
            random_idx = np.random.randint(len(self.indices))
            img_path = self.image_list[self.indices[random_idx]]
            mask_path = self.mask_list[self.indices[random_idx]]
            with rasterio.open(img_path) as src:
                img = src.read(1)
                img = np.expand_dims(img, axis=-1)
                img = tf.image.resize(img, self.image_size)
                img = normalize_image(img)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                mask = np.expand_dims(mask, axis=-1)
                mask = tf.image.resize(mask, self.image_size)
                mask = tf.where(mask > self.threshold, 1, 0)
                mask = remove_small_objects(mask, min_size=self.min_area)
                mask = skeletonize_and_buffer(mask, buffer_size=self.buffer_size)
                if self.augment:
                    img, mask = self.augment_image(img, mask)
                img = img.numpy()
#                 mask = mask.numpy()
                if np.sum(mask) < self.min_area:
                    continue
                X[valid_samples] = img
                y[valid_samples] = np.expand_dims(mask, axis=-1)
                valid_samples += 1

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def plot_images_masks(images, masks, num=2):
    fig, axs = plt.subplots(num, 2, figsize=(10, 5*num))
    for i in range(num):
        axs[i, 0].imshow(images[i, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(masks[i, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Mask')
    plt.tight_layout()
    plt.show()

def main(args):
    """
    Main function to run the data generator script.

    Parameters:
    - args (argparse.Namespace): Parsed command-line arguments.

    The function performs the following steps:
    1. Reads the patch directory and collects image and mask file paths.
    2. Splits the data into training and validation sets.
    3. Initializes the data generators for training and validation.
    4. Retrieves a batch of training and validation data.
    5. Prints the shapes and value ranges of the training and validation batches.
    6. Optionally plots the images and masks from the training batch.
    """
    size = args.size
    min_area = args.min_area
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    patch_dir = args.patch_dir
    threshold = args.threshold

    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patch_dir, f) for f in all_files if 'img' in f]
    train_labels = [os.path.join(patch_dir, f.replace('img', 'lab')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('img', 'lab')))]

    print(f"Total training images: {len(train_images)}")
    print(f"Total training labels: {len(train_labels)}")

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

    image_size = (size, size)
    train_gen = DataGenerator(train_images, train_labels, batch_size=batch_size, image_size=image_size, shuffle=True, min_area=min_area, buffer_size=buffer_size, threshold=threshold)
    val_gen = DataGenerator(val_images, val_labels, batch_size=batch_size, image_size=image_size, shuffle=False, min_area=min_area, buffer_size=buffer_size, threshold=threshold)

    X_batch, y_batch = train_gen[np.random.randint(len(train_gen))]
    X_val_batch, y_val_batch = val_gen[np.random.randint(len(val_gen))]

    print("Training batch shapes:", X_batch.shape, y_batch.shape)
    print("Validation batch shapes:", X_val_batch.shape, y_val_batch.shape)
    print("Max and min in X_batch:", X_batch.max(), X_batch.min())
    print("Max and min in y_batch:", y_batch.max(), y_batch.min())
    print("Max and min in X_val_batch:", X_val_batch.max(), X_val_batch.min())
    print("Max and min in y_val_batch:", y_val_batch.max(), y_val_batch.min())

    if args.plot:
        plot_images_masks(X_batch, y_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Generator Script")
    parser.add_argument('--size', type=int, default=512, help='Size of the patches')
    parser.add_argument('--min_area', type=int, default=100, help='Minimum area for filtering')
    parser.add_argument('--buffer_size', type=int, default=3, help='Buffer size for skeletonization')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for the generator')
    parser.add_argument('--patch_dir', type=str, required=True, help='Directory where patches are stored')
    parser.add_argument('--threshold', type=int, default=30, help='Threshold value for binarizing the mask')
    parser.add_argument('--plot', action='store_true', help='Plot images and masks')
    args = parser.parse_args()
    main(args)





