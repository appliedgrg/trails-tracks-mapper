#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###### USAGE:
# python train.py --patch_dir /path/to/patch_dir --size 512 --filters 32 --learning_rate 0.00003 --batch_size 2 --epochs 50 --checkpoint_dir /media/irro/Irro/CNN_Models/ --min_area 100 --buffer_size 3 --threshold 30


# In[ ]:

import os
import argparse
import tensorflow as tf
from keras_unet.models import custom_unet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_generator import DataGenerator  # Make sure this import matches your data generator script
from utils import iou, iou_thresholded  # Make sure these are defined in your utils
import tensorflow as tf
from scipy.ndimage import label
from skimage.morphology import skeletonize, dilation, square

def iou(y_true, y_pred, smooth=1e-6):
    """
    Compute the Intersection over Union (IoU) between the true and predicted segmentation masks.
    
    Args:
    y_true: the ground truth tensor.
    y_pred: the predicted tensor.
    smooth: a small constant to avoid division by zero.

    Returns:
    IoU score.
    """
    # Flatten the input to convert to 1D
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Compute intersection and union areas
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    # Compute the IoU score
    return (intersection + smooth) / (union + smooth)

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Compute IoU score after applying a threshold to the predicted masks,
    turning probabilities into binary predictions.

    Args:
    y_true: the ground truth tensor.
    y_pred: the predicted probability tensor.
    threshold: segmentation threshold.
    smooth: a small constant to avoid division by zero.

    Returns:
    Thresholded IoU score.
    """
    # Apply threshold to prediction probabilities to create binary mask
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)

    # Calculate IoU score for the binary predictions
    return iou(y_true, y_pred, smooth)

def load_data(patch_dir):
    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patch_dir, f) for f in all_files if 'img' in f]
    train_labels = [os.path.join(patch_dir, f.replace('img', 'lab')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('img', 'lab')))]
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    return train_images, val_images, train_labels, val_labels

def train_model(args):
    # Set input shape
    input_shape = (args.size, args.size, 1)  # Add channel dimension
    print(f"Input shape: {input_shape}")

    # Define the model
    model = custom_unet(
        input_shape=input_shape,
        filters=args.filters,
        use_batch_norm=True,
        dropout=0.3,
        num_classes=1,
        output_activation='sigmoid'
    )

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=[iou, iou_thresholded])

    # Early stopping and learning rate reduction on plateau
    earlystopper = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)

    # Define checkpoint path
    checkpoint_filepath = os.path.join(args.checkpoint_dir, f"trails_tracks_model_epoch_{{epoch:02d}}_valloss_{{val_loss:.2f}}.h5")

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    # Prepare data generators
    train_gen = DataGenerator(
        image_list=args.train_images,
        mask_list=args.train_labels,
        batch_size=args.batch_size,
        image_size=(args.size, args.size),
        shuffle=True,
        min_area=args.min_area,
        buffer_size=args.buffer_size,
        threshold=args.threshold
    )

    val_gen = DataGenerator(
        image_list=args.val_images,
        mask_list=args.val_labels,
        batch_size=args.batch_size,
        image_size=(args.size, args.size),
        shuffle=False,
        min_area=args.min_area,
        buffer_size=args.buffer_size,
        threshold=args.threshold
    )

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        steps_per_epoch=len(train_gen) // args.batch_size,
        validation_steps=len(val_gen) // args.batch_size,
        callbacks=[earlystopper, reduce_lr, checkpoint_callback]
    )

    return history

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
    train_images, val_images, train_labels, val_labels = load_data(args.patch_dir)
    args.train_images = train_images
    args.val_images = val_images
    args.train_labels = train_labels
    args.val_labels = val_labels

    train_model(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net model for segmentation")
    parser.add_argument('--size', type=int, default=512, help='Size of the patches')
    parser.add_argument('--filters', type=int, default=32, help='Number of filters in the first layer')
    parser.add_argument('--learning_rate', type=float, default=0.00003, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for the generator')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--checkpoint_dir', type=str, default='/media/irro/Irro/CNN_Models/', help='Directory to save model checkpoints')
    parser.add_argument('--patch_dir', type=str, required=True, help='Directory where patches are stored')
    parser.add_argument('--min_area', type=int, default=100, help='Minimum area for filtering')
    parser.add_argument('--buffer_size', type=int, default=3, help='Buffer size for skeletonization')
    parser.add_argument('--threshold', type=int, default=30, help='Threshold value for binarizing the mask')
    parser.add_argument('--plot', action='store_true', help='Plot images and masks')
    args = parser.parse_args()
    main(args)

