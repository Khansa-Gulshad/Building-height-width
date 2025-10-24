import os
os.environ['USE_PYGEOS'] = '0'

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.signal import find_peaks
import torch

import google_streetview.api

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import csv

from street_view.segmentation_images import save_all_products

from PIL import Image, ImageFile
from io import BytesIO
import numpy as np
import requests

ImageFile.LOAD_TRUNCATED_IMAGES = True

def prepare_folders(city):
    # Create folder for storing GVI results, sample points and road network if they don't exist yet
    dir_path = os.path.join("results", city, "gvi")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    dir_path = os.path.join("results", city, "points")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join("results", city, "roads")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join("results", city, "sample_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    

def get_models():
    # Load the pretrained AutoImageProcessor from the "facebook/mask2former-swin-large-cityscapes-semantic" model
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pretrained Mask2FormerForUniversalSegmentation model from "facebook/mask2former-swin-large-cityscapes-semantic"
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)
    # Return the processor and model as a tuple
    return processor, model


def segment_images(image, processor, model):
    # Preprocess the image using the image processor
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform a forward pass through the model to obtain the segmentation
    with torch.no_grad():
        # Check if a GPU is available
        if torch.cuda.is_available():
            # Move the inputs to the GPU
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            # Perform the forward pass through the model
            outputs = model(**inputs)
            # Post-process the semantic segmentation outputs using the processor and move the results to CPU
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].to('cpu')
        else:
            # Perform the forward pass through the model
            outputs = model(**inputs)
            # Post-process the semantic segmentation outputs using the processor
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            
    return segmentation


# Based on Matthew Danish code (https://github.com/mrd/vsvi_filter/tree/master)
def run_length_encoding(in_array):
    arr = np.asarray(in_array)
    n = arr.shape[0]
    if n == 0:
        return None, None
    # find change points
    pairwise_unequal = arr[1:] != arr[:-1]
    change_points = np.append(np.where(pairwise_unequal)[0], n - 1)
    run_lengths = np.diff(np.append(-1, change_points))
    return run_lengths, arr[change_points]  # values at run ends (bool if input is bool)

def get_road_pixels_per_column(prediction):
    """
    prediction: HxW labels (torch or numpy). Returns per-column longest vertical run of 'road' (label==0).
    """
    # ensure numpy boolean mask
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    else:
        prediction = np.asarray(prediction)

    road_pixels = (prediction == 0)  # HxW bool
    H, W = road_pixels.shape
    out = np.zeros(W, dtype=np.int32)

    for i in range(W):
        runs, vals = run_length_encoding(road_pixels[:, i])
        if runs is None:
            out[i] = 0
            continue
        # select lengths where the run ends in True (road)
        idx = np.nonzero(vals)[0]
        if idx.size == 0:
            out[i] = 0
        else:
            out[i] = int(runs[idx].max())
    return out

def get_road_centres(prediction, distance=2000, prominence=100):
    # prediction must be numpy
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    else:
        prediction = np.asarray(prediction)

    road_pixels_per_col = get_road_pixels_per_column(prediction)
    peaks, _ = find_peaks(road_pixels_per_col, distance=distance, prominence=prominence)
    return peaks


def find_road_centre(segmentation):
    """
    segmentation: HxW label map (torch or numpy). Returns peak x-indices.
    """
    # ensure numpy
    seg_np = segmentation.detach().cpu().numpy() if torch.is_tensor(segmentation) else np.asarray(segmentation)
    # scale thresholds relative to a 5760x2880 baseline (your original logic)
    distance   = int(2000 * seg_np.shape[1] // 5760)
    prominence = int( 100 * seg_np.shape[0] // 2880)
    centres = get_road_centres(seg_np, distance=distance, prominence=prominence)
    return centres


def crop_panoramic_images_roads(original_width, image, segmentation, road_centre):
    width, height = image.size

    # Find duplicated centres
    duplicated_centres = [centre - original_width for centre in road_centre if centre >= original_width]
            
    # Drop the duplicated centres
    road_centre = [centre for centre in road_centre if centre not in duplicated_centres]

    # Calculate dimensions and offsets
    w4 = int(original_width / 4) # 
    h4 = 0
    hFor43 = height
    w98 = width + (w4 / 2)
    xrapneeded = int(width * 7 / 8)

    images = []
    pickles = []

    # Crop the panoramic image based on road centers
    for centre in road_centre:
        # Wrapped all the way around
        if centre >= w98:
            xlo = int((width - centre) - w4/2)
            cropped_image = image.crop((xlo, h4, xlo + w4, h4 + hFor43))
            cropped_segmentation = segmentation[h4:h4+hFor43, xlo:xlo+w4]
        
        # Image requires assembly of two sides
        elif centre > xrapneeded:
            xlo = int(centre - (w4/2)) # horizontal_offset
            w4_p1 = width - xlo
            w4_p2 = w4 - w4_p1

            # Crop and concatenate image and segmentation
            cropped_image_1 = image.crop((xlo, h4, xlo + w4_p1, h4 + hFor43))
            cropped_image_2 = image.crop((0, h4, w4_p2, h4 + hFor43))

            cropped_image = Image.new(image.mode, (w4, hFor43))
            cropped_image.paste(cropped_image_1, (0, 0))
            cropped_image.paste(cropped_image_2, (w4_p1, 0))

            cropped_segmentation_1 = segmentation[h4:h4+hFor43, xlo:xlo+w4_p1]
            cropped_segmentation_2 = segmentation[h4:h4+hFor43, 0:w4_p2]
            cropped_segmentation = torch.cat((cropped_segmentation_1, cropped_segmentation_2), dim=1)
        
        # Must paste together the two sides of the image
        elif centre < (w4 / 2):
            w4_p1 = int((w4 / 2) - centre)
            xhi = width - w4_p1
            w4_p2 = w4 - w4_p1

            # Crop and concatenate image and segmentation
            cropped_image_1 = image.crop((xhi, h4, xhi + w4_p1, h4 + hFor43))
            cropped_image_2 = image.crop((0, h4, w4_p2, h4 + hFor43))

            cropped_image = Image.new(image.mode, (w4, hFor43))
            cropped_image.paste(cropped_image_1, (0, 0))
            cropped_image.paste(cropped_image_2, (w4_p1, 0))

            cropped_segmentation_1 = segmentation[h4:h4+hFor43, xhi:xhi+w4_p1]
            cropped_segmentation_2 = segmentation[h4:h4+hFor43, 0:w4_p2]
            cropped_segmentation = torch.cat((cropped_segmentation_1, cropped_segmentation_2), dim=1)
            
        # Straightforward crop
        else:
            xlo = int(centre - w4/2)
            cropped_image = image.crop((xlo, h4, xlo + w4, h4 + hFor43))
            cropped_segmentation = segmentation[h4:h4+hFor43, xlo:xlo+w4]
        
        images.append(cropped_image)
        pickles.append(cropped_segmentation)

    return images, pickles


def crop_panoramic_images(image, segmentation):
    width, height = image.size

    w4 = int(width / 4)
    h4 = int(height / 4)
    hFor43 = int(w4 * 3 / 4)

    images = []
    pickles = []

    # Crop the panoramic image based on road centers
    for w in range(4):
        x_begin = w * w4
        x_end = (w + 1) * w4
        cropped_image = image.crop((x_begin, h4, x_end, h4 + hFor43))
        cropped_segmentation = segmentation[h4:h4+hFor43, x_begin:x_end]

        images.append(cropped_image)
        pickles.append(cropped_segmentation)
    
    return images, pickles
    

def process_images(image, cut_by_road_centres, processor, model):
    try:      
        # Get the size of the image
        width, height = image.size

        # Apply the semantic segmentation to the image
        segmentation = segment_images(image, processor, model)
            
        if cut_by_road_centres:
            # Create a widened panorama by wrapping the first 25% of the image onto the right edge
            width, height = image.size
            w4 = int(0.25 * width)
                
            segmentation_25 = segmentation[:, :w4]
            # Concatenate the tensors along the first dimension (rows) to create the widened panorama with the segmentations
            segmentation_road = torch.cat((segmentation, segmentation_25), dim=1)

            cropped_image = image.crop((0, 0, w4, height))
            widened_image = Image.new(image.mode, (width + w4, height))
            widened_image.paste(image, (0, 0))
            widened_image.paste(cropped_image, (width, 0))

            # Find the road centers to determine if the image is suitable for analysis
            road_centre = find_road_centre(segmentation_road)
                
            # Crop the image and its segmentation based on the previously found road centers
            images, pickles = crop_panoramic_images_roads(width, widened_image, segmentation_road, road_centre)
        
        else:
            # Cut panoramic image in 4 equal parts
            # Crop the image and its segmentation based on the previously found road centers
            images, pickles = crop_panoramic_images(image, segmentation)

        return images, pickles, [False, False]

    except:
        # If there was an error while processing the image, set the "error" flag to true and continue with other images
        return None, None, [None, True, True]


# Download images
def download_image(id, geometry, save_sample, city, cut_by_road_centres, access_token, processor, model):
    try:
        params = [{
            'size': '640x640',
            'location': f"{geometry.y},{geometry.x}",
            'heading': 0,
            'fov': '90',
            'key': f"{access_token}",
        }]
        first = google_streetview.api.results(params)
        pano_id = first.metadata[0]['pano_id']  # get once

        panorama_images = [Image.open(requests.get(first.links[0], stream=True).raw)]

        for angle in [90, 180, 270]:
            params = [{
                'size': '640x640',
                'pano': pano_id,
                'heading': angle,
                'fov': '90',
                'key': f"{access_token}",
            }]
            resp = google_streetview.api.results(params)
            panorama_images.append(Image.open(requests.get(resp.links[0], stream=True).raw))

        if len(panorama_images) > 0:
            W = len(panorama_images) * panorama_images[0].width  # 2560
            H = panorama_images[0].height                        # 640
            panorama = Image.new(panorama_images[0].mode, (W, H))
            for i, img in enumerate(panorama_images):
                panorama.paste(img, (i * img.width, 0))

            images, segmentations, result = process_images(panorama, cut_by_road_centres, processor, model)

            if save_sample and images is not None and segmentations is not None:
                for k, (img_k, seg_k) in enumerate(zip(images, segmentations), start=1):
                    image_id_k = f"{pano_id}_{k}"
                    save_all_products(city, image_id_k, img_k, seg_k, out_root=cfg.PROJECT_DIR)

            # result = [GVI, missing=False, error=False] from process_images
            if result is None:
                result = [None, True, True]
        else:
            result = [None, True, False]

    except Exception as e:
        # optional: print(e)
        result = [None, True, True]
        pano_id = None  # ensure defined

    # Build CSV row: [id, x, y, pano_id, GVI, missing, error]
    row = [id, geometry.x, geometry.y, pano_id] + result
    return row

def download_images_for_points(gdf, access_token, max_workers, cut_by_road_centres, city, file_name):
    # Get image processing models
    processor, model = get_models()

    # Prepare CSV file path
    csv_file = f"points-{file_name}.csv"
    csv_path = os.path.join("results", city, csv_file)

    # Check if the CSV file exists and chose the correct editing mode
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'

    # Create a lock object for thread safety
    results = []
    lock = threading.Lock()
    
    # Open the CSV file in append mode with newline=''
    with open(csv_path, mode, newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(["id", "x", "y", "pano_id", "missing", "error"])
        
        # Create a ThreadPoolExecutor to process images concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Iterate over the rows in the GeoDataFrame
            for _, row in gdf.iterrows():
                try:
                    # Submit a download_image task to the executor
                    futures.append(executor.submit(download_image, row["id"], row["geometry"], row["save_sample"], city, cut_by_road_centres, access_token, processor, model))
                except Exception as e:
                    print(f"Exception occurred for row {row['id']}: {str(e)}")
            
            # Process the completed futures using tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading images"):
                # Retrieve the result of the completed future
                image_result = future.result()
				

                # Acquire the lock before appending to results and writing to the CSV file
                with lock:
                    results.append(image_result)
                    writer.writerow(image_result)
		

    # Return the processed image results
    return results
