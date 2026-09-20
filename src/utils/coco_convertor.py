"""
coco_converter.py
─────────────────
Converts BRISC 2025 segmentation masks into COCO format JSON.
For each image: derives bounding box from mask, converts to COCO format,
and saves everything into a single JSON file for Grounding DINO training.
"""

import os
import json
from src.utils.data_utils import load_binary_mask, mask_to_bbox, bbox_to_coco, parse_filename

CATEGORIES = [
    {"id": 1, "name": "glioma"},
    {"id": 2, "name": "meningioma"},
    {"id": 3, "name": "pituitary"},
]

CATEGORY_NAME_TO_ID = {
    "glioma"     : 1,
    "meningioma" : 2,
    "pituitary"  : 3,
}

def convert_to_coco(images_dir, masks_dir, output_path):
    """
    Convert BRISC masks to COCO format JSON.
    
    Args:
        images_dir  : path to folder containing .jpg MRI images
        masks_dir   : path to folder containing .png mask files
        output_path : where to save the output JSON file
    """
    coco = {
        "images"      : [],
        "annotations" : [],
        "categories"  : CATEGORIES,
    }

    image_id      = 1
    annotation_id = 1

    image_files = sorted(os.listdir(images_dir))

    for filename in image_files:
        if not filename.endswith(".jpg"):
            continue

        meta = parse_filename(filename)
        if meta is None:
            continue
        
        class_name = meta["class_name"]
        
        if class_name == "healthy":
            continue

        mask_filename = filename.replace(".jpg", ".png")
        mask_path = os.path.join(masks_dir, mask_filename)      

        binary_mask = load_binary_mask(mask_path)
        bbox = mask_to_bbox(binary_mask)

        if bbox is None:
            continue    

        coco["images"].append({
                "id"       : image_id,
                "file_name": filename,
                "width"    : 512,
                "height"   : 512,
            })

        coco_bbox, area = bbox_to_coco(bbox, 512, 512)
            
        coco["annotations"].append({
            "id"         : annotation_id,
            "image_id"   : image_id,
            "category_id": CATEGORY_NAME_TO_ID[class_name],
            "bbox"       : coco_bbox,
            "area"       : area,
        })

        image_id      += 1
        annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    
    print(f"Saved {len(coco['images'])} images to {output_path}")