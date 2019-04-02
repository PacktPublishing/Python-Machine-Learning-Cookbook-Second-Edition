#pip install --user opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10
#to run: python feature_detector.py flowers.jpg
import sys

import cv2
import numpy as np

# Load input image -- 'table.jpg'
input_file = sys.argv[1]
img = cv2.imread(input_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(img_gray, None)

img_sift = np.copy(img)
cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Input image', img)
cv2.imshow('SIFT features', img_sift)
cv2.waitKey()
