#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy, Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
from collections import namedtuple

PIDConfig = namedtuple('PIDConfig', 'KP, KI, KD')

class PIDController:
  control_params = ["theta"] #, "v"]
  param_configs = {"theta": PIDConfig(0.02, 0.05, 0.0025)}
  #KP = 0.002
  #KI = 0.0001
  #KD = 0.00022
  DRIVE_SPEED = .8
  Y_REFERENCE = 0

  def __init__(self, sub_topic, pub_topic, ref_sub_topic):
    # Create the publisher and subscriber
    self.sub = rospy.Subscriber(sub_topic, PoseStamped, self.point_cb)
    self.ref_sub = rospy.Subscriber(ref_sub_topic, PoseStamped, self.reference_cb)
    self.reset_sub = rospy.Subscriber("/pid/reset", Empty, self.reset_cb)
    
    self.pub = rospy.Publisher(pub_topic, AckermannDriveStamped)
    self.param_confirm_pub = rospy.Publisher("/exp_tool/hparam_confirm", Empty, queue_size=1)

    self.pose = None
    self.integral_sum = 0
    self.prev_time = None 
    self.prev_error = None
    self.prev_x = 0
    self.ref = None
     
    self.drive_speed = self.DRIVE_SPEED
    # subscribe to joystick to recognize custom inputs
    self.joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_cb)
    self.image_sub = rospy.Subscriber('/ip/roi', Image, self.image_cb)

    self.show_next_image = False

    # tf listener to get map to base_link transforms
    self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    self.bridge = CvBridge()
  
  def reset_cb(self, msg):
    for controller in self.control_params:
      config = self.param_configs[controller]
      t = {} # Create dict to set data into named tuple
      for (i, param) in enumerate(config._fields):
        data = rospy.get_param('/pid/{}/{}'.format(controller, param), config[i])
        t[param] = data
      self.param_configs[controller] = config._replace(**t)
    rospy.logerr("Refreshed all controller parameters! And reset integral term.")
    self.drive_speed = rospy.get_param("/pid/v", self.DRIVE_SPEED)
    self.integral_sum = 0.0 
    self.prev_error = 0
    self.param_confirm_pub.publish(Empty())
    self.cycle_theta(0.1)
    
  def reference_cb(self, msg):
    self.ref = msg
    # rospy.logerr(str(self.ref))
    cur_time = msg.header.stamp.to_sec()
    # rospy.logerr("new x pos: " + str(self.ref))
    self.cycle_theta(cur_time)

  def point_cb(self, msg):
    self.cycle_theta(msg.header.stamp.to_sec())

  def cycle_theta(self, cur_time):
    if (not self.ref):
      x = 0
    else:
      ref_transformed =  self.transform_to_local(self.ref)
      rospy.logerr(ref_transformed)
      x = ref_transformed.pose.position.y

    if x == -1:
      x = self.prev_x

    if self.prev_time is None: 
      self.prev_time = cur_time

    dt = cur_time - self.prev_time + 1e-1000
    dt = max(0.0001, min(dt, 0.5))
    cur_error = x - self.Y_REFERENCE
    rospy.logerr("Curr error " + str(cur_error) + " x: " + str(x))

    if self.prev_error is None:
      self.prev_error = cur_error

    self.integral_sum += cur_error * dt
    d_error = float(cur_error - self.prev_error) / dt
    
    cfg = self.param_configs['theta']
    control = cfg.KP * cur_error + cfg.KI * self.integral_sum + cfg.KD * d_error 

    self.prev_time  = cur_time
    self.prev_error = cur_error
    self.prev_x = x

    msg = AckermannDriveStamped() 
    msg.drive.speed = self.drive_speed
    msg.drive.steering_angle = control
    self.pub.publish(msg)
    # rospy.logerr(str(control) + " " + str(self.integral_sum))

  def joy_cb(self, msg):
    # buttons=[A, B, X, Y, LB, RB, Back, Start, Logitech, Left joy, Right joy]
    if msg.buttons[6]: # Back is pressed : reset error
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

  def transform_to_local(self, msg):
    transform = self.tf_buffer.lookup_transform("ta_laser", 
                                                msg.header.frame_id, 
                                                rospy.Time(0))
    pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, transform)
    return pose_transformed
      
    

if __name__ == '__main__':
  img_sub_topic = None # The image topic to be subscribed to.
  pub_topic = None # The topic to publish filtered images to

  rospy.init_node('apply_filter', anonymous=True)

  # Populate params with values passed by launch file
  sub_topic = rospy.get_param('point_sub_topic')
  pub_topic = rospy.get_param('ackermann_pub_topic') 
  ref_topic = rospy.get_param('ref_topic') 

  # Create a Filter object and pass it the loaded parameters
  PIDController(sub_topic, pub_topic, ref_topic)

  rospy.spin()
    
