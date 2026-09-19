"""
data_utils.py
─────────────
Core data loading utilities for the BRISC 2025 dataset.

What this file does:
  - Parses BRISC filenames to extract metadata
  - Loads MRI images and segmentation masks
  - Converts noisy masks to clean binary (threshold=127)
  - Derives bounding boxes from binary masks
  - Builds a pandas DataFrame of the entire dataset

Usage:
  from src.utils.data_utils import load_dataset, load_binary_mask, mask_to_bbox
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


# ── Constants ───────────────────────────────────────────────────────────────

MASK_THRESHOLD = 127   # pixels above this = tumor, below = background

CLASS_MAP = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "healthy",
}

PLANE_MAP = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}

# Text prompts we send to Grounding DINO for each class
PROMPT_MAP = {
    "glioma"     : "glioma tumor",
    "meningioma" : "meningioma tumor",
    "pituitary"  : "pituitary tumor",
    "healthy"    : "healthy brain",
}


# ── Filename parsing ─────────────────────────────────────────────────────────

def parse_filename(filename):
    """
    Parse a BRISC filename and return a dict of metadata.

    Example input:
        brisc2025_train_00001_gl_ax_t1.jpg

    Example output:
        {
            "case_id"    : "00001",
            "class_code" : "gl",
            "class_name" : "glioma",
            "plane"      : "ax",
            "plane_name" : "axial",
            "sequence"   : "t1",
            "prompt"     : "glioma tumor",
            "filename"   : "brisc2025_train_00001_gl_ax_t1.jpg",
        }
    """
    name  = os.path.splitext(filename)[0]   # remove .jpg or .png
    parts = name.split("_")

    # Guard against unexpected filenames
    if len(parts) < 6:
        return None

    class_code = parts[3]
    class_name = CLASS_MAP.get(class_code, class_code)

    return {
        "case_id"    : parts[2],
        "class_code" : class_code,
        "class_name" : class_name,
        "plane"      : parts[4],
        "plane_name" : PLANE_MAP.get(parts[4], parts[4]),
        "sequence"   : parts[5],
        "prompt"     : PROMPT_MAP.get(class_name, "brain tumor"),
        "filename"   : filename,
    }


# ── Image and mask loading ───────────────────────────────────────────────────

def load_image(image_path):
    """
    Load an MRI image as a numpy RGB array.

    Why RGB even though MRI is grayscale?
    Grounding DINO and MedSAM both expect 3-channel input.
    Converting to RGB duplicates the single channel 3 times.
    The values stay the same — just the shape changes.

    Returns: numpy array of shape (H, W, 3), dtype uint8
    """
    return np.array(Image.open(image_path).convert("RGB"))


def load_binary_mask(mask_path):
    """
    Load a segmentation mask and convert to clean binary.

    Why do we threshold at 127?
    BRISC masks should be 0 (background) or 255 (tumor).
    Due to PNG compression artifacts, edge pixels get values
    like 1, 2, 3 ... 253, 254.
    Thresholding at 127 cleanly separates background from tumor.

    Returns: numpy array of shape (H, W), dtype uint8, values in {0, 1}
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > MASK_THRESHOLD).astype(np.uint8)


# ── Bounding box derivation ──────────────────────────────────────────────────

def mask_to_bbox(binary_mask):
    """
    Derive a bounding box from a binary segmentation mask.

    How it works:
      1. Find all rows that contain at least one tumor pixel
      2. Find all columns that contain at least one tumor pixel
      3. The box corners are the outermost of those rows and columns

    Example:
      mask:          rows with tumor:   cols with tumor:
      0 0 0 0 0      row 1 ✓            col 1 ✓
      0 1 1 0 0      row 2 ✓            col 2 ✓
      0 1 1 1 0      row 3 ✓            col 3 ✓
      0 0 0 0 0
      
      → bbox = [x_min=1, y_min=1, x_max=3, y_max=3]

    Args:
        binary_mask: numpy array with values 0 and 1

    Returns:
        [x_min, y_min, x_max, y_max] or None if no tumor pixels
    """
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)

    if not rows.any():
        return None   # healthy image — no tumor

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def bbox_to_coco(bbox, image_width, image_height):
    """
    Convert [x_min, y_min, x_max, y_max] to COCO format.

    COCO format uses [x_min, y_min, width, height]
    We also compute the area.

    Why COCO format?
    MMDetection (used to train Grounding DINO) expects COCO format.

    Returns: (coco_bbox, area)
        coco_bbox = [x_min, y_min, width, height]
        area      = width * height
    """
    x_min, y_min, x_max, y_max = bbox
    width  = x_max - x_min
    height = y_max - y_min
    area   = width * height

    return [x_min, y_min, width, height], area


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(images_dir, masks_dir):
    """
    Build a DataFrame describing every image in a split.

    For each image we record:
      - All metadata from the filename
      - Full paths to image and mask files
      - The derived bounding box
      - Whether it has a tumor

    Args:
        images_dir : path to folder containing .jpg MRI images
        masks_dir  : path to folder containing .png mask files

    Returns:
        pandas DataFrame, one row per image
    """
    records    = []
    image_files = sorted(os.listdir(images_dir))

    for filename in image_files:
        if not filename.endswith(".jpg"):
            continue

        # Parse filename metadata
        meta = parse_filename(filename)
        if meta is None:
            continue

        # Build file paths
        image_path = os.path.join(images_dir, filename)
        mask_name  = filename.replace(".jpg", ".png")
        mask_path  = os.path.join(masks_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        # Derive bounding box from mask
        binary_mask = load_binary_mask(mask_path)
        bbox        = mask_to_bbox(binary_mask)
        has_tumor   = bbox is not None

        # Compute tumor size statistics
        tumor_pixels = int(np.sum(binary_mask))
        total_pixels = binary_mask.size
        tumor_ratio  = tumor_pixels / total_pixels

        record = {
            **meta,
            "image_path"  : image_path,
            "mask_path"   : mask_path,
            "bbox"        : bbox,           # [x_min, y_min, x_max, y_max]
            "has_tumor"   : has_tumor,
            "tumor_pixels": tumor_pixels,
            "tumor_ratio" : tumor_ratio,
        }

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            record["bbox_width"]  = x_max - x_min
            record["bbox_height"] = y_max - y_min
        else:
            record["bbox_width"]  = 0
            record["bbox_height"] = 0

        records.append(record)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} images from {images_dir}")
    return df
