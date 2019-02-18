#!/usr/bin/env python

import rospy
import math
import numpy as np
import tf.transformations
from geometry_msgs.msg import Point, Pose, Quaternion

# THESE FUNCTIONS MAY OR MAY NOT BE HELPFUL IN YOUR IMPLEMENTATION
# IMPLEMENT/USE AS YOU PLEASE

def point_to_pose(point):
  p = Point(point[0], point[1], 0)
  o = angle_to_quaternion(point[2])
  return Pose(p,o)

def rotation(theta):
  c, s = np.cos(theta), np.sin(theta)
  return np.array([[c, -s],[s, c]]).reshape(2,2)

def angle_to_quaternion(angle):
  return Quaternion(*tf.transformations.quaternion_from_euler(0,0,angle))

def quaternion_to_angle(q):
  roll, pitch, yaw = quaternion_to_euler_angle(q)
  return math.radians(yaw)

def quaternion_to_euler_angle(quat):
  # Based on wiki conversion algorithm
  w, x, y, z = quat.w, quat.x, quat.y, quat.z
  ysqr = y * y
  
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + ysqr)
  X = math.degrees(math.atan2(t0, t1))
  
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  Y = math.degrees(math.asin(t2))
  
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (ysqr + z * z)
  Z = math.degrees(math.atan2(t3, t4))
  
  return X, Y, Z

def map_to_world(poses,map_info):
  angle = quaternion_to_angle(map_info.origin.orientation)
  s, c = np.sin(angle), np.cos(angle)
  
  xs = np.copy(poses[:,0])
  poses[:,0] = xs * c - poses[:,1] * s
  poses[:,1] = xs * s + poses[:,1] * c
  
  #scale everything except angles
  poses[:,:2] *= float(map_info.resolution)
  
  # translate all
  poses[:,0] += map_info.origin.position.x
  poses[:,1] += map_info.origin.position.y
  poses[:,2] += angle

def world_to_map(poses, map_info):
  angle = -quaternion_to_angle(map_info.origin.orientation)
  s, c = np.sin(angle), np.cos(angle)

  # translate everything first
  poses[:,0] -= map_info.origin.position.x
  poses[:,1] -= map_info.origin.position.y
  poses[:,2] += angle

  # scale down the positions but not the angle
  poses[:,:2] /= float(map_info.resolution)

  # rotate into map frame
  xs = np.copy(poses[:,0]) 
  poses[:,0] = xs * c - poses[:,1] * s
  poses[:,1] = xs * s + poses[:,1] * c

