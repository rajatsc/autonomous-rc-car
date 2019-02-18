#!/usr/bin/env python

import time
import sys
import utils
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessing:
  LOWER_BLUE = np.array([90, 110, 120])
  UPPER_BLUE = np.array([100, 255, 255])
  LOWER_RED1 = np.array([0, 230, 120])
  UPPER_RED1 = np.array([2, 255, 255])
  #LOWER_RED2 = np.array([170, 110, 120])
  #UPPER_RED2 = np.array([179, 255, 255])
  LOWER_RED2 = np.array([160, 110, 160])
  UPPER_RED2 = np.array([179, 255, 255])
  #LOWER_GREEN = np.array([29, 110, 80])
  #UPPER_GREEN = np.array([55, 255, 255])

  LOWER_GREEN = np.array([35, 100, 50])
  UPPER_GREEN = np.array([75, 255, 255])
  #ROI_HEIGHT = 200
  #ROI_OFFSET = 70
  #ROI_HEIGHT = 130
  #ROI_OFFSET = 50
  ROI_CENTER_R = 3
  ROI_CENTER_COLOR = [0, 255, 0]  # bgr
  MIN_PIXEL_THRESH = 10
  DOWNSAMPLE = 1.0

  def __init__(self, img_sub_topic, pub_topic, color, roi_height, roi_offset, feedforward=False):
    self.bridge = CvBridge()
    self.color = color
    self.ROI_HEIGHT = roi_height
    self.ROI_OFFSET = roi_offset
    self.feedforward = feedforward

    self.pub = rospy.Publisher(pub_topic, PointStamped, queue_size=1)
    self.sub = rospy.Subscriber(img_sub_topic, Image, self.process_image_cb)
    
    self.roi_pub = rospy.Publisher('/ip/roi', Image, queue_size=1)
    self.mask_pub = rospy.Publisher('/ip/mask', Image, queue_size=1)
    self.img_pub = rospy.Publisher('/ip/path_img', Image, queue_size=1)
    self.coordinates = None
    self.weights = None
  
    self.times = []

  def process_image_cb(self, msg):
    try:
      start = time.time()

      bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
      path_image, mask = self.get_path_image(bgr_image)
      x, y = self.get_roi_center_naiive(mask)
      
      colored_roi = path_image[-self.ROI_HEIGHT-self.ROI_OFFSET:-self.ROI_OFFSET, :]
      self.color_roi_center(colored_roi, x, y)
      
      # publish images to show in rviz
      self.img_pub.publish(self.bridge.cv2_to_imgmsg(path_image.astype(np.uint8), encoding='bgr8'))
      self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask.astype(np.uint8), encoding='mono8'))
      self.roi_pub.publish(self.bridge.cv2_to_imgmsg(colored_roi.astype(np.uint8), encoding='bgr8'))
      #cv2.imwrite("/home/nvidia/roi.png", colored_roi)
      #rospy.logerr("roi saved")
 
      # publish the ROI center
      msg = PointStamped()
      msg.point = Point(x,y,0)
      msg.header.stamp = rospy.get_rostime()
      self.pub.publish(msg)

      self.times.append(time.time() - start)
      #rospy.logerr(len(self.times))
    except CvBridgeError as e:
      rospy.logerr(e)

  def get_path_image(self, bgr_image):
    # 1. downsample image
    if self.DOWNSAMPLE < 1.0:
      bgr_image = cv2.resize(bgr_image, (0,0), fx=self.DOWNSAMPLE, fy=self.DOWNSAMPLE)
    # 2. blur
    blur = cv2.GaussianBlur(bgr_image, (5,5), 0)
    # get hsv
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 3. get mask
    if self.color == 'blue':
      mask = cv2.inRange(hsv_image, self.LOWER_BLUE, self.UPPER_BLUE)
    elif self.color == 'red':
      mask1 = cv2.inRange(hsv_image, self.LOWER_RED1, self.UPPER_RED1)
      mask2 = cv2.inRange(hsv_image, self.LOWER_RED2, self.UPPER_RED2)
      mask = cv2.bitwise_or(mask2, mask1)
    elif self.color == 'green':
      mask = cv2.inRange(hsv_image, self.LOWER_GREEN, self.UPPER_GREEN)

    # 4. get masked image
    res = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

    return blur, mask

  def get_roi_center_naiive(self, mask):
    roi = mask[-self.ROI_HEIGHT-self.ROI_OFFSET:-self.ROI_OFFSET, :]

    if self.coordinates is None:
      width = roi.shape[1]
      height = roi.shape[0]
      coordinates = np.zeros((width*height, 2), dtype=np.int)
      coordinates[:,0] = np.tile(np.arange(width), height)  # xs
      coordinates[:,1] = np.repeat(np.arange(height), width)  # ys
      self.coordinates = coordinates.reshape((height, width, 2))
      #self.weights = 1.0 / np.arange(1, self.ROI_HEIGHT + 1)
      self.weights = np.abs(np.arange(-width/2, width/2+1), dtype=np.float)

    # get the list of nonzero coordinate (returned shape: num_nonzero, 2)
    nonzero_coords = self.coordinates[roi > 0]
    if nonzero_coords.shape[0] < self.MIN_PIXEL_THRESH:
      return (-1, -1)

    center_y = np.sum(nonzero_coords[:,1]) / nonzero_coords.shape[0]
    if not feedforward:
      center_x = np.sum(nonzero_coords[:,0]) / nonzero_coords.shape[0]
    else:
      # weight coords with lower y values higher
      weights = self.weights[nonzero_coords[:,0]]
      weights /= np.sum(weights)
      weighted_xs = nonzero_coords[:,0] * weights
      center_x = weighted_xs.sum()
      # rospy.logerr(weighted_xs)
      # rospy.logerr(weights)
      # rospy.logerr(center_x)
    return (center_x, center_y)

  def color_roi_center(self, roi, x, y):
    # highlight center
    roi[y-self.ROI_CENTER_R:y+self.ROI_CENTER_R,
        x-self.ROI_CENTER_R:x+self.ROI_CENTER_R] = self.ROI_CENTER_COLOR

  def benchmark(self):
    mean = np.mean(self.times)
    variance = np.var(self.times)
    rospy.logerr("TIME MEAN: {}".format(mean))
    rospy.logerr("TIME VAR: {}".format(variance))

if __name__ == '__main__':
  rospy.init_node('apply_filter', anonymous=True)
  
  img_sub_topic = rospy.get_param('sub_topic')
  pub_topic = rospy.get_param('pub_topic')
  color = rospy.get_param('color', 'blue')
  roi_height = rospy.get_param('roi_height', 130)
  roi_offset = rospy.get_param('roi_offset', 50)
  
  feedforward = rospy.get_param('feedforward', False)

  rospy.logerr((roi_height, roi_offset, color, feedforward))
  ip = ImageProcessing(img_sub_topic, pub_topic, color, roi_height, roi_offset, feedforward)
  rospy.on_shutdown(ip.benchmark)
  rospy.spin()
