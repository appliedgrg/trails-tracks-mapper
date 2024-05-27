#!/usr/bin/env python
# coding: utf-8

# In[1]:


# python prediction.py --input_tif /path/to/input.tif --model_name /path/to/model.h5 --patch_size 512 --overlap_size 256 --filters 32 --visualize


# In[1]:


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[22]:


import os
import numpy as np
import argparse
import rasterio
from rasterio.windows import Window
from keras_unet.models import custom_unet
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import skeletonize, dilation, square

# Constants
BLOCK_SIZE = 256

# Helper Functions
def adjust_window(window, max_width, max_height):
    col_off, row_off, width, height = window.flatten()
    if col_off + width > max_width:
        width = max_width - col_off
    if row_off + height > max_height:
        height = max_height - row_off
    return Window(col_off, row_off, width, height)

def pad_to_shape(array, target_shape):
    diff_height = target_shape[0] - array.shape[0]
    diff_width = target_shape[1] - array.shape[1]
    padded_array = np.pad(array, ((0, diff_height), (0, diff_width), (0, 0)), 'constant')
    return padded_array

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def predict_patch_optimized(model, patch, vis=False):
    if patch.max() != patch.min():
        patch_norm = normalize_image(patch)
    else:
        patch_norm = patch
    
    num_bands = patch.shape[2] if len(patch.shape) == 3 else 1
    patch_norm = patch_norm.reshape(1, patch_norm.shape[0], patch_norm.shape[1], num_bands)
    
    pred_patch = model.predict(patch_norm)

    if vis:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(patch_norm.squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title('Original Patch')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_patch.squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title('Predicted Patch')
        plt.axis('off')
        plt.show()
    
    pred_patch = (pred_patch * 100).squeeze().astype(np.uint8)
    
    return pred_patch

def process_block(src, x_start, x_end, y_start, y_end, patch_size, stride, model, vis=False):
    pred_accumulator = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)
    counts = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint16)

    for i in tqdm(range(y_start, y_end, stride), desc=f"Processing y range {y_start} to {y_end}"):
        for j in range(x_start, x_end, stride):
            window = Window(j, i, patch_size, patch_size)
            window = adjust_window(window, x_end, y_end)
            patch = src.read(window=window)
            patch = np.moveaxis(patch, 0, -1)

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = pad_to_shape(patch, (patch_size, patch_size))

            pred_patch = predict_patch_optimized(model, patch, vis)

            col_off, row_off, width, height = map(int, window.flatten())
            pred_patch = pred_patch[:height, :width]
            accumulator_indices = (slice(row_off - y_start, row_off + height - y_start), 
                                   slice(col_off - x_start, col_off + width - x_start))
            pred_accumulator[accumulator_indices] = np.maximum(pred_accumulator[accumulator_indices], pred_patch)
            counts[accumulator_indices] += 1

    return pred_accumulator

def plot_predictions(original_patch, prediction):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_patch.squeeze(), cmap='gray')
    axs[0].set_title('Original Patch')
    im = axs[1].imshow(prediction, cmap='jet', vmin=0, vmax=100)
    axs[1].set_title('Predicted Probabilities')
    fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()

def predict_tif_optimized(model, path, patch_size, overlap_size, vis, model_name):
    stride = patch_size - overlap_size
    with rasterio.open(path) as src:
        original_height, original_width = src.shape
        pred_accumulator = np.zeros((original_height, original_width), dtype=np.uint8)
        counts = np.zeros((original_height, original_width), dtype=np.uint16)

        for i in tqdm(range(0, original_height, stride)):
            for j in range(0, original_width, stride):
                window = Window(j, i, patch_size, patch_size)
                window = adjust_window(window, original_width, original_height)
                patch = src.read(window=window)
                patch = np.moveaxis(patch, 0, -1)

                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = pad_to_shape(patch, (patch_size, patch_size))

                pred_patch = predict_patch_optimized(model, patch, vis)
                col_off, row_off, width, height = map(int, window.flatten())
                pred_patch = pred_patch[:height, :width]
                pred_accumulator[row_off:row_off+height, col_off:col_off+width] = np.maximum(pred_accumulator[row_off:row_off+height, col_off:col_off+width], pred_patch)
                counts[row_off:row_off+height, col_off:col_off+width] += 1

        final_pred = pred_accumulator

        output_path = path[:-4] + f'_{model_name}_{overlap_size}.tif'
        profile = src.profile.copy()
        profile.update(dtype='uint8', nodata=0, compress='LZW', tiled=True, blockxsize=BLOCK_SIZE, blockysize=BLOCK_SIZE)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(final_pred[np.newaxis, :, :])

    return output_path

# Main Function
def main(args):
    model = load_model(args.model_name, custom_objects={'iou': iou, 'iou_thresholded': iou_thresholded})

    # Extract and print the input shape
    input_shape = model.layers[0].input_shape
    print("Input shape of the model:", input_shape)

    patch_size = input_shape[0][1]
    print("Image size:", patch_size)

    output_path = predict_tif_optimized(
        model,
        args.input_tif,
        patch_size,
        args.overlap_size,
        args.visualize
    )

    print(f'Saved final prediction to: {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict segmentation using U-Net model")
    parser.add_argument('--input_tif', type=str, required=True, help='Path to the input GeoTIFF file')
    parser.add_argument('--model_name', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--overlap_size', type=int, default=256, help='Overlap size between patches')
    parser.add_argument('--filters', type=int, default=32, help='Number of filters in the first layer')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    args = parser.parse_args()
    main(args)


# In[23]:


# model_name = '/home/irro/trails-tracks-mapper/models/Human_DTMnorm50_512_byCNN_5ep.h5'
# # model_name = '/home/irro/trails-tracks-mapper/models/Human_DTMnorm50_512_byCNN_5ep.h5'

# model = load_model(model_name, custom_objects={'iou': iou, 'iou_thresholded': iou_thresholded})

# # Extract and print the input shape
# input_shape = model.layers[0].input_shape
# print("Input shape of the model:", input_shape)

# patch_size = input_shape[0][1]
# print("Image size:", patch_size)


# In[24]:


# path = '/media/irro/Irro/HumanFootprint/TEST/FEN_aerial_normDTM50cm.tif'

# predict_tif_optimized(model, path, patch_size, int(patch_size*0.5), True, os.path.basename(model_name)[:-3])


# In[ ]:




