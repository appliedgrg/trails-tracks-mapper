#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import geopandas as gpd
import argparse
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol

def generate_and_save_patches(shp, tif_path, mask_path, size, base_dir, idnumber_start=0):
    '''Function for cropping patches from the beginning. The same procedure for every tile:
    1) shapefile with points to dataframe with square coordinate WITHIN the tile boundaries
    2) reading tif_path and label layer to numpy array & filtering their boundaries to make correspond to each other
    with their shapes and covered area
    3) Vizualization just to check that everythin is right
    4) Cropping patches using dataframe and saving them to the base_dir
    
    Input
    shp:          shapefile with (usually random) points. We'll build our training patches around them
    tif_path:     full path to original tif file with your data
    mask:         Label layer with EVERY pixel having a value of 1
                  for object (e.g. line footprint in case of line mapping)
                  and 0 for the rest area (e.g. lake, forest, peatland)
                  Label should be of the same resolution as "tif"
    size:         size of your output patches (e.g. 256 pix * 256 pix) 
    base_dir:     directory to save our patches!
    idnumber:     init number of patch we use when saving it (e.g. patch_661.png means idnumber = 661)
    '''
    
    tif_array, mask_array, bounds, transform = align_and_crop(tif_path, mask_path)

    # Ensure the output directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Process shapefile to get patch locations
    df = shp_to_patches(shp, tif_path, size)
    print(f'\nNumber of points within the shapefile: {len(df)}')

    # Initialize idnumber from the start parameter
    idnumber = idnumber_start
    num_patches = 0  # Counter for the number of patches generated

    # Loop through dataframe to extract patches
    for index, row in df.iterrows():
        # Use the pixel coordinates directly for cropping
        patch = tif_array[:, int(row.py_min):int(row.py_max), int(row.px_min):int(row.px_max)]
        patch_l = mask_array[:, int(row.py_min):int(row.py_max), int(row.px_min):int(row.px_max)]
        # print(f"For ID {idnumber} shapes are: patch {patch.shape} and label {patch_l.shape}.")

        if patch.shape[1:] > (size, size) or patch_l.shape[1:] > (size, size):
            # Crop to the desired size
            patch = patch[:, :size, :size]
            patch_l = patch_l[:, :size, :size]
        elif patch.shape[1:] != (size, size) or patch_l.shape[1:] != (size, size):
            print(f"Skipping tile {idnumber} due to improper shape: {patch.shape}, {patch_l.shape}.")
            continue  # Skip this patch due to incorrect size

        # Prepare filenames for saving
        filename_p = os.path.join(base_dir, f'{idnumber}_img.tif')
        filename_l = os.path.join(base_dir, f'{idnumber}_lab.tif')

        # Save the patches as georeferenced TIFFs using metadata from original TIFF
        save_geotiff(filename_p, patch, tif_path, row.px_min, row.py_min, size)
        save_geotiff(filename_l, patch_l, tif_path, row.px_min, row.py_min, size)

        # Increment counters for patch ID and total number of patches
        num_patches += 1
        idnumber += 1

    # Log how many patches were processed
    print(f'Total patches generated and saved: {num_patches}')

def rowcol_to_pixelbounds(cx, cy, transform, half_size):
    '''
    Converts center coordinates to pixel bounds.
    '''
    px, py = rowcol(transform, cx, cy)
    return (px - half_size, py - half_size, px + half_size, py + half_size)

def align_and_crop(tif_path, mask_path):
    '''Align TIFF and mask layers and return arrays with same geographic and pixel coverage.'''
    
    with rasterio.open(tif_path) as tif, rasterio.open(mask_path) as mask:
        # Calculate intersection bounds
        left = max(tif.bounds.left, mask.bounds.left)
        bottom = max(tif.bounds.bottom, mask.bounds.bottom)
        right = min(tif.bounds.right, mask.bounds.right)
        top = min(tif.bounds.top, mask.bounds.top)
        
        # Convert intersection bounds to window for each layer
        tif_window = from_bounds(left, bottom, right, top, tif.transform)
        mask_window = from_bounds(left, bottom, right, top, mask.transform)

        # Crop each layer based on the intersection window
        tif_array = tif.read(window=tif_window)
        mask_array = mask.read(window=mask_window)

        # Ensure same size if slight discrepancy exists after cropping
        min_height = min(tif_array.shape[1], mask_array.shape[1])
        min_width = min(tif_array.shape[2], mask_array.shape[2])
        tif_array = tif_array[:, :min_height, :min_width]
        mask_array = mask_array[:, :min_height, :min_width]

        return tif_array, mask_array, (left, bottom, right, top), tif.transform
    
def save_geotiff(filename, data, reference_tif, x_offset, y_offset, patch_size):
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        # Calculate new origin (top-left corner) for the patch
        new_origin_x, _ = src.transform * (x_offset, 0)  # For x coordinate
        _, new_origin_y = src.transform * (0, y_offset)  # For y coordinate

        # Create new transform for the patch
        new_transform = rasterio.transform.from_origin(new_origin_x, new_origin_y, src.res[0], src.res[1])
        meta.update({
            'driver': 'GTiff',
            'height': patch_size,
            'width': patch_size,
            'transform': new_transform,
            'count': data.shape[0]  # Assumes 'data' has shape [bands, rows, cols]
        })
        
        with rasterio.open(filename, 'w', **meta) as dst:
            # If data is single-banded but in 3D format, select the first band.
            if data.ndim == 3 and data.shape[0] == 1:
                dst.write(data[0], 1)  # Write the first (and only) band data as 2D.
            elif data.ndim == 3:
                dst.write(data)  # Multi-band data, write as is.
            else:
                raise ValueError("Unexpected data shape for writing: ", data.shape)

def shp_to_patches(shp, tif_path, size=256):
    '''
    Transforming our shapefile with points to the pandas dataframe with patches (squares) around them!
    & Filtering shp by boundaries of the tif file.
    
    Input
    shp:          shapefile with (usually random) points. We'll build our training patches around them
    tif_path:     full path to original tif file with your data
    size:         size of your output patches (e.g. 256 pix * 256 pix)
    
    Output
    result:  pandas dataframe with 'image_path' of your original tif, 
             boundaries and center coordinates (minx, miny - non geographic, tile_xmin, tile_ymin - geographical)
             and your label (only for object detection).
    '''
    with rasterio.open(tif_path) as tif:
        # Use the raster's bounds and resolution
        left, bottom, right, top = tif.bounds
        resolution = tif.res[0]  # Assuming square pixels
        transform = tif.transform  # Get the affine transform of the raster
        
        # Read shapefile
        gdf = gpd.read_file(shp).to_crs(tif.crs)  # Ensure same CRS

        # Filtering shapefile by TIFF boundaries using spatial indexing
        gdf_filtered = gdf.cx[left:right, bottom:top]
        
        # Prepare DataFrame for patches with geographic coordinates
        half_width = (size * resolution) / 2  # Half the patch size in map units
        gdf_filtered["image_path"] = tif_path
        gdf_filtered["center_x"] = gdf_filtered.geometry.x
        gdf_filtered["center_y"] = gdf_filtered.geometry.y
        gdf_filtered["tile_xmin"] = gdf_filtered["center_x"] - half_width
        gdf_filtered["tile_xmax"] = gdf_filtered["center_x"] + half_width
        gdf_filtered["tile_ymin"] = gdf_filtered["center_y"] - half_width
        gdf_filtered["tile_ymax"] = gdf_filtered["center_y"] + half_width
        
        # Initialize columns for pixel coordinates
        gdf_filtered['px_min'] = None
        gdf_filtered['px_max'] = None
        gdf_filtered['py_min'] = None
        gdf_filtered['py_max'] = None

        # Convert geographic coordinates to pixel coordinates
        for index, row in gdf_filtered.iterrows():
            py_min, px_min = rowcol(transform, row['tile_xmin'], row['tile_ymin'])
            py_max, px_max = rowcol(transform, row['tile_xmax'], row['tile_ymax'])
            gdf_filtered.at[index, 'px_min'] = px_min
            gdf_filtered.at[index, 'px_max'] = px_max
            gdf_filtered.at[index, 'py_max'] = py_min   ### works
            gdf_filtered.at[index, 'py_min'] = py_max   ### works

        # Ensure the patches are within the image bounds
        gdf_filtered = gdf_filtered[
            (gdf_filtered['px_min'] >= 0) & (gdf_filtered['px_max'] <= tif.width) &
            (gdf_filtered['py_min'] >= 0) & (gdf_filtered['py_max'] <= tif.height)
        ]

        result = gdf_filtered[['image_path', 'center_x', 'center_y', 'px_min', 'py_min', 'px_max', 'py_max']].copy()
        print('Check Dataframe\n*****', result.head(1))

    return result
                
def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument('--train_tif', type=str, required=True, help='Path to the training TIFF file')
    parser.add_argument('--mask_tif', type=str, required=True, help='Path to the mask TIFF file')
    parser.add_argument('--patch_dir', type=str, required=True, help='Directory to save patches')
    parser.add_argument('--shp', type=str, required=True, help='Path to the shapefile')
    parser.add_argument('--size', type=int, default=512, help='Patch size')
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.exists(args.patch_dir):
        os.makedirs(args.patch_dir)

    generate_and_save_patches(args.shp, args.train_tif, args.mask_tif, args.size, args.patch_dir, idnumber_start=0)

if __name__ == "__main__":
    main()

