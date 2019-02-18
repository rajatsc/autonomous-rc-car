#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy, Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError

class PersonFollower:
  IM_TARGET_DEPTH = .75
  IM_TARGET_CENTER = 216
  angle_KP = 0.0019
  angle_KI = 0.0001
  angle_KD = 0.00035

  range_KP = 2.0
  range_KI = 0.01
  range_KD = 0.09

  MAX_SPEED = 0.8

  def __init__(self, point_sub_topic, range_sub_topic, pub_topic):
    # Create the publisher and subscriber
    self.point_sub = rospy.Subscriber(point_sub_topic, PointStamped, self.point_cb)
    self.range_sub = rospy.Subscriber(range_sub_topic, LaserScan, self.range_cb)
    self.pub = rospy.Publisher(pub_topic, AckermannDriveStamped)

    self.integral_sum = 0
    self.prev_time = None 
    self.prev_error = None
    self.prev_x = self.IM_TARGET_DEPTH
    self.speed = 0
     
    self.angle_integral_sum = 0
    self.angle_prev_time = None 
    self.angle_prev_error = None
    self.angle_prev_x = self.IM_TARGET_CENTER
    self.steering_angle = 0
    self.prev_angle = 0

    # subscribe to joystick to recognize custom inputs
    self.joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_cb)
    self.image_sub = rospy.Subscriber('/ip/roi', Image, self.image_cb)
    self.show_next_image = False
    self.bridge = CvBridge()

  def point_cb(self, msg):
    x = msg.point.x

    if x == -1:
      x = self.angle_prev_x

    cur_time = msg.header.stamp.to_sec()

    if self.angle_prev_time is None: 
      self.angle_prev_time = cur_time

    dt = cur_time - self.angle_prev_time
    cur_error = self.IM_TARGET_CENTER - x

    if self.angle_prev_error is None:
      self.angle_prev_error = cur_error

    self.angle_integral_sum += cur_error * dt
    d_error = float(cur_error - self.angle_prev_error) / dt
    
    self.steering_angle = self.angle_KP * cur_error + self.angle_KI * self.angle_integral_sum + self.angle_KD * d_error 

    self.angle_prev_time  = cur_time
    self.angle_prev_error = cur_error
    self.angle_prev_x = x

    msg = AckermannDriveStamped() 
    msg.drive.speed = self.speed
    if self.speed < 0:
      self.steering_angle *= -1
    self.prev_angle = 0.7 * self.prev_angle + 0.3 * self.steering_angle
    msg.drive.steering_angle = self.prev_angle
    self.pub.publish(msg)
    #rospy.logerr("steering angle: " + str(self.steering_angle))
 
  def range_cb(self, msg):
    # find min range
    amin = msg.angle_min
    amax = msg.angle_max
    awidth = amax - amin
    
    ranges = np.array(msg.ranges)
    ranges = ranges[210:-210]
    ranges = ranges[np.isfinite(ranges)]
    # get 50 min elements
    step = 35
    ignore = 20
    filtered = ranges[ranges > 0.007]
    indices = filtered.argpartition(step+ignore)[ignore:step+ignore]
    x = np.median(filtered[indices])
    rospy.logerr("=========> " + str(x) + "<==========")

    if x == -1:
      #x = self.prev_x
      x = 0 

    cur_time = msg.header.stamp.to_sec()

    if self.prev_time is None: 
      self.prev_time = cur_time

    dt = cur_time - self.prev_time
    cur_error = x - self.IM_TARGET_DEPTH

    if self.prev_error is None:
      self.prev_error = cur_error

    self.integral_sum += cur_error * dt
    d_error = float(cur_error - self.prev_error) / dt
    
    control = self.range_KP * cur_error + self.range_KI * self.integral_sum + self.range_KD * d_error 

    self.prev_time  = cur_time
    self.prev_error = cur_error
    self.prev_x = x

    if control > self.MAX_SPEED:
      control = self.MAX_SPEED 
    elif control < -self.MAX_SPEED:
      control = -self.MAX_SPEED
    self.speed = control
    rospy.logerr("speed: " + str(self.speed))

  def joy_cb(self, msg):
    # buttons=[A, B, X, Y, LB, RB, Back, Start, Logitech, Left joy, Right joy]
    if msg.buttons[3]: # Y is pressed : reset error
      self.integral_sum = 0.0
      rospy.logerr('Y pressed') 
    if msg.buttons[1]: # B is pressed : print out image from camera
      self.show_next_image = True
      rospy.logerr('B pressed')

  def image_cb(self, msg):
    if self.show_next_image:
      rospy.logerr('Printing image')
      self.show_next_image = False
      bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
      cv2.imshow('Processed camera image', bgr_image)
      cv2.waitKey(0) 

if __name__ == '__main__':
  img_sub_topic = None # The image topic to be subscribed to.
  pub_topic = None # The topic to publish filtered images to

  rospy.init_node('apply_filter', anonymous=True)

  # Populate params with values passed by launch file
  point_sub_topic = rospy.get_param('point_sub_topic')
  range_sub_topic = rospy.get_param('range_sub_topic')
  pub_topic = rospy.get_param('ackermann_pub_topic') 

  # Create a Filter object and pass it the loaded parameters
  PersonFollower(point_sub_topic, range_sub_topic, pub_topic)

  rospy.spin()
    
