#!/usr/bin/env python3
# scripts/make_sample.py
"""
Generates a synthetic test image for the example pipelines.
Run: python scripts/make_sample.py
"""
import numpy as np
import cv2
from pathlib import Path

Path("examples").mkdir(exist_ok=True)

# 640×480 dark background
img = np.full((480, 640, 3), 30, dtype=np.uint8)

# Draw some shapes to create detectable contours
cv2.rectangle(img, (50,  50),  (180, 150), (200, 220, 255), -1)
cv2.rectangle(img, (220, 80),  (380, 200), (255, 180, 100), -1)
cv2.circle   (img, (480, 120), 70,         (100, 255, 180), -1)
cv2.ellipse  (img, (150, 320), (90, 50), 30, 0, 360, (255, 100, 200), -1)
cv2.rectangle(img, (300, 280), (580, 420), (180, 255, 255), -1)

# Add a gradient background hint
for y in range(480):
    alpha = y / 480.0
    img[y] = np.clip(img[y].astype(float) + alpha * 20, 0, 255).astype(np.uint8)

# Add some noise
noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

cv2.imwrite("examples/sample.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("✓ Generated examples/sample.jpg (640×480)")
