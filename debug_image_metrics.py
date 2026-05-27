#!/usr/bin/env python
"""Debug script to analyze image metrics for specific photos."""

from pathlib import Path
from config_loader import load_config
from qc_photo import (
    _histogram_entropy,
    _histogram_entropy,
    _low_content_patch_ratio,
    _mean_saturation,
    _std_dev_gray,
)
from qc_video import _laplacian_variance_gray, _mean_brightness_bgr
import cv2
import numpy as np

import sys
test_images = sys.argv[1:]

config = load_config()

# Test image paths - these should be actual paths from your sorted output
test_images = [
    # Add paths to the problematic images if available locally
    # For now we'll just print config
]

print("Current QC Config:")
print(f"  blur_threshold: {config['blur_threshold']}")
print(f"  contrast_threshold: {config['contrast_threshold']}")
print(f"  content_variance_threshold: {config['content_variance_threshold']}")
print(f"  content_variance_reject_ratio: {config['content_variance_reject_ratio']}")
print(f"  saturation_threshold: {config['saturation_threshold']}")
print(f"  saturation_reject_ratio: {config['saturation_reject_ratio']}")
print(f"  histogram_entropy_threshold: {config['histogram_entropy_threshold']}")
print(f"  histogram_entropy_reject: {config['histogram_entropy_reject']}")

if test_images:
    for img_path in test_images:
        if not Path(img_path).exists():
            continue
        print(f"\nAnalyzing: {img_path}")
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print("  Cannot read image")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blur = _laplacian_variance_gray(gray)
        contrast = _std_dev_gray(gray)
        content_ratio = _low_content_patch_ratio(gray, config['content_variance_threshold'])
        saturation = _mean_saturation(frame)
        entropy = _histogram_entropy(gray)
        brightness = _mean_brightness_bgr(frame)
        
        print(f"  Blur (Laplacian var): {blur:.2f} (threshold: {config['blur_threshold']})")
        print(f"  Contrast (std dev): {contrast:.2f} (threshold: {config['contrast_threshold']})")
        print(f"  Content low-var patches: {100*content_ratio:.1f}% (reject if >= {100*config['content_variance_reject_ratio']:.1f}%)")
        print(f"  Saturation mean: {saturation:.4f} (threshold: {config['saturation_threshold']})")
        print(f"  Entropy: {entropy:.2f} (threshold: {config['histogram_entropy_threshold']})")
        print(f"  Brightness: {brightness:.2f}")
