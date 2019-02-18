#!/usr/bin/env python

import sys
import utils
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge, CvBridgeError

class ImageTracking:
  ROI_HEIGHT = 130
  ROI_OFFSET = 50
  ROI_CENTER_R = 3
  ROI_CENTER_COLOR = [0, 255, 0]  # bgr
  MIN_PIXEL_THRESH = 10

  def __init__(self, img_sub_topic, pub_topic, color):
    # Create the publisher and subscriber
    self.pub = rospy.Publisher(pub_topic, PointStamped, queue_size=1)
    self.sub = rospy.Subscriber(img_sub_topic, Image, self.process_image_cb)
    # Create a CvBridge object for converting sensor_msgs/Image into numpy arrays (and vice-versa)
    #		http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    self.bridge = CvBridge()
    
    self.roi_pub = rospy.Publisher('/ip/roi', Image, queue_size=1)
    self.mask_pub = rospy.Publisher('/ip/mask', Image, queue_size=1)
    self.img_pub = rospy.Publisher('/ip/path_img', Image, queue_size=1)
    self.coordinates = None
    self.bbox = None

  def process_image_cb(self, msg):
    try:
      # get and show raw image
      bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
      if self.bbox == None:
        self.tracker = cv2.Tracker_create('MIL')
        self.bbox = cv2.selectROI(bgr_image, False)
        self.tracker.init(bgr_image, self.bbox)
      else:
        ok, self.bbox = tracker.update(bgr_image)
      x = self.bbox[0]
      y = self.bbox[1]
      w = self.bbox[2]
      h = self.bbox[3]
      center_x = int(x + w/2)
      center_y = int(y + h/2)
 
      # publish the ROI center
      msg = PointStamped()
      msg.point = Point(center_x,center_y,0)
      msg.header.stamp = rospy.get_rostime()
      self.pub.publish(msg)
    except CvBridgeError as e:
      rospy.logerr(e)


  def get_roi_center_naiive(self, path_image):
    roi = path_image[-self.ROI_HEIGHT-self.ROI_OFFSET:-self.ROI_OFFSET, :]

    if self.coordinates is None:
      width = roi.shape[1]
      height = roi.shape[0]
      coordinates = np.zeros((width*height, 2), dtype=np.int)
      coordinates[:,0] = np.tile(np.arange(width), height)  # xs
      coordinates[:,1] = np.repeat(np.arange(height), width)  # ys
      self.coordinates = coordinates.reshape((height, width, 2))

    # get the list of nonzero "b" channel coordinate (returned shape: num_nonzero, 2)
    nonzero_coords = self.coordinates[roi[:,:,0] > 0]
    if nonzero_coords.shape[0] < self.MIN_PIXEL_THRESH:
      return (roi, -1, -1)

    center_x = np.sum(nonzero_coords[:,0]) / nonzero_coords.shape[0]
    center_y = np.sum(nonzero_coords[:,1]) / nonzero_coords.shape[0]
    # highlight center
    roi[center_y-self.ROI_CENTER_R:center_y+self.ROI_CENTER_R,
        center_x-self.ROI_CENTER_R:center_x+self.ROI_CENTER_R] = self.ROI_CENTER_COLOR

    #keypoints = self.detector.detect(roi)
    #im_with_keypoints = cv2.drawKeypoints(roi, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('keypoints', im_with_keypoints)

    return (roi, center_x, center_y)


if __name__ == '__main__':
  rospy.init_node('apply_filter', anonymous=True)
  
  img_sub_topic = rospy.get_param('sub_topic')
  pub_topic = rospy.get_param('pub_topic')
  color = rospy.get_param('color')

  ImageTracking(img_sub_topic, pub_topic, color)
  rospy.spin()
