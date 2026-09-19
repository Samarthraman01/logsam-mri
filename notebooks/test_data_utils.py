"""
Quick test to verify data_utils.py works correctly.
Run this on MacBook to check the code — no GPU needed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_utils import (
    parse_filename,
    load_binary_mask,
    mask_to_bbox,
    bbox_to_coco,
    CLASS_MAP,
    PROMPT_MAP,
)
import numpy as np

print("Testing data_utils.py")
print("=" * 50)

# Test 1 — filename parsing
print("\nTest 1: parse_filename")
test_file = "brisc2025_train_00001_gl_ax_t1.jpg"
result    = parse_filename(test_file)
print(f"  Input : {test_file}")
print(f"  Output: {result}")
assert result["class_name"] == "glioma",  "class name wrong"
assert result["prompt"]     == "glioma tumor", "prompt wrong"
assert result["case_id"]    == "00001", "case id wrong"
print("  ✅ PASSED")

# Test 2 — mask to bbox
print("\nTest 2: mask_to_bbox")
fake_mask         = np.zeros((10, 10), dtype=np.uint8)
fake_mask[3:7, 2:8] = 1   # tumor region
bbox              = mask_to_bbox(fake_mask)
print(f"  Mask tumor region: rows 3-6, cols 2-7")
print(f"  Derived bbox     : {bbox}")
assert bbox == [2, 3, 7, 6], f"Expected [2,3,7,6] got {bbox}"
print("  ✅ PASSED")

# Test 3 — empty mask
print("\nTest 3: mask_to_bbox on empty mask (healthy)")
empty_mask = np.zeros((10, 10), dtype=np.uint8)
bbox       = mask_to_bbox(empty_mask)
print(f"  Result: {bbox}  (should be None)")
assert bbox is None, "Should return None for empty mask"
print("  ✅ PASSED")

# Test 4 — bbox to COCO
print("\nTest 4: bbox_to_coco")
coco_bbox, area = bbox_to_coco([10, 20, 50, 80], 512, 512)
print(f"  Input    : [10, 20, 50, 80]")
print(f"  COCO     : {coco_bbox}  (x,y,w,h)")
print(f"  Area     : {area}")
assert coco_bbox == [10, 20, 40, 60], f"Wrong: {coco_bbox}"
assert area      == 2400, f"Wrong area: {area}"
print("  ✅ PASSED")

print("\n" + "=" * 50)
print("All tests passed ✅")
