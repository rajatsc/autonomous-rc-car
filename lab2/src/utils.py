#!/usr/bin/env python

import sys
import cv2
import numpy as np

ROI_CENTER_R = 3
ROI_CENTER_COLOR = [0, 255, 0]  # bgr
MIN_PIXEL_THRESH = 10

# Params
# bgr_image: the raw image in BGR
# color_bound: tuple (lower, upper) to define the target color range (in HSV)
# COLOR BOUND IN OPENCV IS IN RANGE (0-180,0-255,0-255)
def get_path_image(bgr_image, color_bound):
  # downsample image
  bgr_image = cv2.resize(bgr_image, (0,0), fx=0.8, fy=0.8)
  hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
  #hsv_image = cv2.GaussianBlur(hsv_image, (7,7), 0)
  #cv2.imshow('raw bgr', bgr_image)
  #cv2.imshow('raw hsv', hsv_image)

  # get blurred binary image
  mask = cv2.inRange(hsv_image, color_bound[0], color_bound[1])
  res = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
  blur = cv2.GaussianBlur(res, (5,5), 0)
  edge = cv2.Canny(blur, 100, 200)
  #cv2.imshow('mask', mask)
  #cv2.imshow('res', res)
  #cv2.imshow('blurred binary', blur)
  #cv2.imshow('edge', edge)

  return blur

def get_roi_center_naiive(path_image, roi_height, roi_offset=0):
  roi = path_image[-roi_height-roi_offset:-roi_offset, :]

  width = roi.shape[1]
  height = roi.shape[0]
  coordinates = np.zeros((width*height, 2), dtype=np.int)
  coordinates[:,0] = np.tile(np.arange(width), height)  # xs
  coordinates[:,1] = np.repeat(np.arange(height), width)  # ys
  coordinates = coordinates.reshape((height, width, 2))

  # get the list of nonzero "b" channel coordinate (returned shape: num_nonzero, 2)
  nonzero_coords = coordinates[roi[:,:,0] > 0]
  if nonzero_coords.shape[0] < MIN_PIXEL_THRESH:
    return (roi, -1, -1)

  center_x = np.sum(nonzero_coords[:,0]) / nonzero_coords.shape[0]
  center_y = np.sum(nonzero_coords[:,1]) / nonzero_coords.shape[0]
  # highlight center
  roi[center_y-ROI_CENTER_R:center_y+ROI_CENTER_R,
      center_x-ROI_CENTER_R:center_x+ROI_CENTER_R] = ROI_CENTER_COLOR

  #keypoints = self.detector.detect(roi)
  #im_with_keypoints = cv2.drawKeypoints(roi, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  #cv2.imshow('keypoints', im_with_keypoints)

  return (roi, center_x, center_y)


if __name__ == '__main__':
  # Usage: python utils.py path/to/image.png
  bgr_image = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

  color_bound = (np.array([90, 130, 150]), np.array([100, 255, 255]))

  image = get_path_image(bgr_image, color_bound)
  roi, x, y = get_roi_center_naiive(image, 100)

  cv2.imshow('path', image)
  cv2.imshow('ROI', roi)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
