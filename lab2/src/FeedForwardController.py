#!/usr/bin/env python
import os
import rospy
import numpy as np
import cv2
#from scipy import signal
from matplotlib import pyplot as plt
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError


class FeedForwardController:
  REDUCTION_FACTOR=0.2
  ROI_HEIGHT = 130 #75
  ROI_OFFSET = 50 #5
  
  def __init__(self, sub_topic, pub_topic, template_path): 
    self.bridge = CvBridge()
    #self.template_images = []
    self.template_coords = []
    self.template_controls = []
    self.template_names = []
    self.template_path = template_path
    self.load_templates(template_path) 

    #self.sub = rospy.Subscriber(sub_topic, Image, self.image_cb)
    self.sub = rospy.Subscriber(sub_topic, PointStamped, self.coord_cb)
    self.pub = rospy.Publisher(pub_topic, AckermannDriveStamped)
    self.template_pub = rospy.Publisher('/feedforward/best_template', Image)
    self.prev_controls = (0,0)
    #print self.template_controls
    #print (len(self.template_controls), len(self.template_images))
    #print self.template_images[0].shape
    print 'done init'

  def load_templates(self, template_path):
    temp_names = os.listdir(template_path) 
    for file_name in temp_names:
      name, ext = os.path.splitext(file_name) 
      if ext != '.npy':
        continue
      #speed, steering_angle = name.split('_') 
      temp_num, speed, steering_angle = name.split('_') 
      speed = float(speed)
      steering_angle = float(steering_angle)
      coord = np.load(template_path + '/' + file_name)
      #img = np.load(template_path + '/' + file_name)
      #sm_img = cv2.resize(img, (0,0), fx=self.REDUCTION_FACTOR, fy=self.REDUCTION_FACTOR)
      #sm_img_roi = sm_img[-self.ROI_HEIGHT:, :]
      #sm_img_roi = sm_img[-self.ROI_HEIGHT-self.ROI_OFFSET:-self.ROI_OFFSET, :]
      #self.template_images.append(sm_img_roi)
      self.template_coords.append(coord)
      self.template_controls.append((speed, steering_angle))
      self.template_names.append(name)

  def coord_cb(self, msg):
    x, y = msg.point.x, msg.point.y
    best_dist = np.sqrt((x - self.template_coords[0][0]) ** 2 + (y - self.template_coords[0][1]) ** 2)
    best_controls = self.template_controls[0]
    best_template_name = ''
    if x == -1 or y == -1:
      best_controls = self.prev_controls
    else:
      for i, (speed, steering_angle) in enumerate(self.template_controls):
        dist = np.sqrt((x - self.template_coords[i][0]) ** 2 + (y - self.template_coords[i][1]) ** 2)
        #rospy.logerr((steering_angle, dist))
        if dist < best_dist:
          best_dist = dist
          best_controls = (speed, steering_angle)
          best_center = (self.template_coords[i][0], self.template_coords[i][1])
          best_template_name = self.template_names[i]
      #rospy.logerr("best controls: {} {}".format(*best_controls))
      #rospy.logerr("im center           {} {}".format(x, y))
      #rospy.logerr("best control center {} {}".format(*best_center))
    # publish the controls
    
    self.prev_controls = best_controls
    msg = AckermannDriveStamped()
    msg.drive.speed = best_controls[0]
    msg.drive.steering_angle = best_controls[1]
    self.pub.publish(msg)

    #if best_template_name != '':
    #  temp_img = cv2.imread(self.template_path + '/' + best_template_name + '.png')
    #  cv2.imwrite("/home/nvidia/best_temp.png", temp_img)
    #  rospy.logerr("temp saved")
    #  self.template_pub.publish(self.bridge.cv2_to_imgmsg(temp_img.astype(np.uint8), encoding='bgr8'))
      
  def image_cb(self, msg):
    try:
      mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
      # controls corresponding to the best template (one with the best score)
      best_controls = (0,0)
      best_score = 0
      # the x coordinate of the best score
      best_score_x = -1
      best_i = -1

      
      roi = mask[-self.ROI_HEIGHT:, :]
      #roi = mask[-self.ROI_HEIGHT-self.ROI_OFFSET:-self.ROI_OFFSET, :]
      print "mask:{}".format(roi.shape)
      for i, (speed, steering_angle) in enumerate(self.template_controls):
        print "image:{}".format(self.template_images[i].shape)
        cons = signal.convolve2d(roi, self.template_images[i], mode='valid')
        score_x = np.argmax(cons)
        score = cons[0, score_x]
        print "score: {}, x_val: {}".format(score, score_x)
        
       # print "temp{}:{}".format(i, self.template_images[i].shape)
       # print "cons:{}".format(cons.shape)
       # print "score_x:{}".format(score_x)
        
        if score > best_score:
          best_score = score
          best_score_x = score_x 
          best_controls = (speed, steering_angle)
          best_i = i
      if best_score == 0:
        best_controls = self.prev_controls
      self.prev_controls = best_controls

      print "best: temp{}{},x={}".format(best_i, best_controls, best_score_x)
      
      # publish the controls
      msg = AckermannDriveStamped()
      msg.drive.speed = best_controls[0]
      msg.drive.steering_angle = best_controls[1]
      self.pub.publish(msg)

      # publish best template
      temp_img = self.bridge.cv2_to_imgmsg(self.template_images[best_i].astype(np.uint8), encoding='mono8')
      self.template_pub.publish(temp_img)
    except CvBridgeError as e:
      rospy.logerr(e)

if __name__ == '__main__':
  rospy.init_node('feed_forward_controller', anonymous=True)
  pub_topic = rospy.get_param('ackermann_pub_topic') 
  #FeedForwardController('/ip/mask', pub_topic, '/home/johan/catkin_ws/src/lab2/templates')
  #FeedForwardController('/ip/line_center', pub_topic, '/home/johan/catkin_ws/src/lab2/templates')
  #FeedForwardController('/ip/line_center', pub_topic, '/home/arielin/nethome/grad/cse490r/catkin_ws/src/lab2/templates')
  
  # working version
  #FeedForwardController('/ip/line_center', pub_topic, '/home/nvidia/catkin_ws/src/lab2/templates')

  # writeup templates
  FeedForwardController('/ip/line_center', pub_topic, '/home/nvidia/catkin_ws/src/lab2/bigger_templates')
  rospy.spin()
