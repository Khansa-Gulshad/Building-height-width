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

from street_view.segmentation_images import  (
    save_full_color, save_three_color, remap_to_three, save_full_overlay, save_rgb,
    save_three_class_mask, save_three_class_npz   # <-- add these two
)

from PIL import Image, ImageFile
from io import BytesIO
import numpy as np
import requests

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- PATHS ----------
BASE_DIR = "/users/project1/pt01183/Building-height-width"

def _city_dir(city: str) -> str:
    return os.path.join(BASE_DIR, city)
	
def prepare_folders(city: str):
    base = os.path.join(cfg.PROJECT_DIR, cfg.city_to_dir(city))
    for sub in ["seg", "seg_3class", "seg_3class_vis", "seg_full_vis", "seg_full_overlay", "sample_images", "save_rgb", "save_three_class_npz"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    

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
    # Convert input array to a NumPy array
    image_array = np.asarray(in_array)
    length = len(image_array)
    if length == 0: 
        # Return None values if the array is empty
        return (None, None, None)
    else:
        # Calculate run lengths and change points in the array
        pairwise_unequal = image_array[1:] != image_array[:-1]
        change_points = np.append(np.where(pairwise_unequal), length - 1)   # must include last element posi
        run_lengths = np.diff(np.append(-1, change_points))       # run lengths
        return(run_lengths, image_array[change_points])

def get_road_pixels_per_column(prediction):
    # Check which pixels in the prediction array correspond to roads (label 0)
    road_pixels = prediction == 0.0
    road_pixels_per_col = np.zeros(road_pixels.shape[1])
    
    for i in range(road_pixels.shape[1]):
        # Encode the road pixels in each column and calculate the maximum run length
        run_lengths, values = run_length_encoding(road_pixels[:,i])
        road_pixels_per_col[i] = run_lengths[values.nonzero()].max(initial=0)
    return road_pixels_per_col

def get_road_centres(prediction, distance=2000, prominence=100):
    # Get the road pixels per column in the prediction
    road_pixels_per_col = get_road_pixels_per_column(prediction)

    # Find peaks in the road_pixels_per_col array based on distance and prominence criteria
    peaks, _ = find_peaks(road_pixels_per_col, distance=distance, prominence=prominence)
    
    return peaks


def find_road_centre(segmentation):
    # Calculate distance and prominence thresholds based on the segmentation shape
	distance = int(2000 * segmentation.shape[1] // 5760)
	prominence = int(100 * segmentation.shape[0] // 2880)
	
    # Find road centers based on the segmentation, distance, and prominence thresholds
	centres = get_road_centres(segmentation, distance=distance, prominence=prominence)
	
	return centres


def crop_panoramic_images_roads(original_width, image, segmentation, road_centre):
    width, height = image.size

    # Find duplicated centres
    duplicated_centres = [centre - original_width for centre in road_centre if centre >= original_width]
            
    # Drop the duplicated centres
    road_centre = [centre for centre in road_centre if centre not in duplicated_centres]

    # Calculate dimensions and offsets
    w4 = int(width / 4) # 
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
        return None, None, [True, True]


# Download images
def download_image(id, geometry, save_sample, city, cut_by_road_centres, access_token, processor, model, fov, pitch):
    try:
        # First request by location to get pano_id + the 0° tile
        q0 = [{
            'size': '640x640',
            'location': f"{geometry.y},{geometry.x}",
            'heading': 0,
            'pitch': str(pitch),
            'fov': str(fov),
            'key': f"{access_token}",
        }]
        first = google_streetview.api.results(q0)
        pano_id = first.metadata[0]['pano_id']

        # Now fetch the same pano at the other headings
        headings = [0, 90, 180, 270]
        tiles = []
        # include the first image we already have
        tiles.append(Image.open(requests.get(first.links[0], stream=True).raw))

        for h in headings[1:]:
            q = [{
                'size': '640x640',
                'pano': pano_id,
                'heading': h,
                'pitch': str(pitch),
                'fov': str(fov),
                'key': f"{access_token}",
            }]
            r = google_streetview.api.results(q)
            tiles.append(Image.open(requests.get(r.links[0], stream=True).raw))

        # Segment and save each tile
        if save_sample:
            for k, img_k in enumerate(tiles, start=1):
                image_id_k = f"{pano_id}_{k}"

                # 1) RGB
                save_rgb(city, image_id_k, img_k)

                # 2) segmentation (per-tile, 640x640 in / out)
                seg_k = segment_images(img_k, processor, model)
                if hasattr(seg_k, "detach"):
                    seg_np = seg_k.detach().cpu().numpy().astype(np.uint8)
                else:
                    seg_np = np.asarray(seg_k, dtype=np.uint8)

                # 3) full-class colorized PNG
                save_full_color(city, image_id_k, seg_np)

                # 4) 3-class: color + npz
                mask3 = remap_to_three(seg_np)          # 0 ground, 1 building, 2 sky
                save_three_color(city, image_id_k, mask3)
                save_three_class_npz(city, image_id_k, mask3)

                # 5) overlay
                save_full_overlay(city, image_id_k, np.array(img_k), seg_np)

        flags = [False, False]  # missing, error

    except Exception:
        pano_id = None
        flags = [True, True]

    # CSV row: id, x, y, pano_id, missing, error
    return [id, geometry.x, geometry.y, pano_id] + flags

def download_images_for_points(gdf, access_token, max_workers, cut_by_road_centres, city, file_name, fov=90, pitch=25):
    processor, model = get_models()

    # ensure city/<points> dir exists (consistent with your structure)
    city_dir = _city_dir(city)  # BASE_DIR/<City>
    points_dir = os.path.join(city_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    # CSV path: /users/.../Building-height-width/<City>/points/points-<file_name>.csv
    csv_path = os.path.join(points_dir, f"points-{file_name}.csv")
    file_exists = os.path.exists(csv_path)
    mode = "a" if file_exists else "w"

    results = []
    lock = threading.Lock()

    with open(csv_path, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # no GVI/IAI/OSI in header, but include pano_id per your request
            writer.writerow(["id", "x", "y", "pano_id", "missing", "error"])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, row in gdf.iterrows():
                try:
                    futures.append(executor.submit(
						download_image,
                		row["id"], row["geometry"], row.get("save_sample", True),
                		city, False,  # force no road-centre cropping
                		access_token, processor, model, fov, pitch
            		))
                except Exception as e:
                    print(f"Exception scheduling row {row['id']}: {e}")

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
                row_out = future.result()
                with lock:
                    results.append(row_out)
                    writer.writerow(row_out)

    return results
