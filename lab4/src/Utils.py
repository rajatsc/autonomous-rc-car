#!/usr/bin/env python

import rospy
import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped,Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import heapq
import math
import csv
import matplotlib.pyplot as plt

from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped

def wait_for_time():
  while rospy.Time().now().to_sec() == 0:
    pass

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw

def posestamped_to_config(posestamped):
  x = posestamped.pose.position.x
  y = posestamped.pose.position.y
  th = quaternion_to_angle(posestamped.pose.orientation)
  return [x, y, th]

def pose_to_config(pose):
  x = pose.position.x
  y = pose.position.y
  th = quaternion_to_angle(pose.orientation)
  return [x, y, th]

def make_header(frame_id, stamp=None):
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

def config_to_pose(config):
    pose = Pose()
    pose.position.x = config[0]
    pose.position.y = config[1]
    pose.orientation = angle_to_quaternion(config[2])
    return pose

def config_to_posestamped(config):
    pose = PoseStamped()
    pose.header = make_header('map')
    pose.pose.position.x = config[0]
    pose.pose.position.y = config[1]
    pose.pose.orientation = angle_to_quaternion(config[2])
    return pose

def plan_to_posearray(plan):
  nodes = PoseArray()
  nodes.header = make_header('map')
  nodes.poses = map(config_to_pose, plan)
  return nodes

def our_world_to_map(config, map_info):
    # equivalent to map_to_grid(world_to_map(poses))
    # operates in place
    pose = np.array(config)
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    pose[0] -= map_info.origin.position.x
    pose[1] -= map_info.origin.position.y

    # scale
    pose[:2] *= (1.0/float(scale))
    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(pose[0])
    pose[0] = c*pose[0] - s*pose[1]
    pose[1] = s*temp    + c*pose[1]
    pose[2] += angle
    return pose

def our_map_to_world(config, map_info):
  pose = np.array(config)

  angle = quaternion_to_angle(map_info.origin.orientation)
  s, c = np.sin(angle), np.cos(angle)
  
  temp = np.copy(pose[0])
  pose[0] = temp * c - pose[1] * s
  pose[1] = temp * s + pose[1] * c
  
  #scale everything except angles
  pose[:2] *= float(map_info.resolution)
  
  # translate all
  pose[0] += map_info.origin.position.x
  pose[1] += map_info.origin.position.y
  pose[2] += angle
  
  return pose

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


def world_to_map(pose, map_info):
    # equivalent to map_to_grid(world_to_map(pose))
    # operates in place
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)
    config = [0.0,0.0,0.0]
    # translation
    config[0] = (1.0/float(scale))*(pose[0] - map_info.origin.position.x)
    config[1] = (1.0/float(scale))*(pose[1] - map_info.origin.position.y)
    config[2] = pose[2]

    temp = np.copy(config[0])
    config[0] = int(c*config[0] - s*config[1])
    config[1] = int(s*temp       + c*config[1])
    config[2] += angle
    
    return config
      
def map_to_world(poses, map_info):
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


def euclidian_distance(a, b):
  return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# CSV:
#   x,y
#   1,1
#   2,5
#   ...
# Return a list of configs [[x, y, 0], [x, y, 0]] in world frame
def load_csv_to_configs(file_name, csv_mode, map_info):
  configs = []
  with open(file_name, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      # ignore title row
      if row[0] == 'x':
        continue

      config = [float(row[0]), float(row[1]), 0.0]
      # pixel to world
      if csv_mode == "pixel":
        config[1] = map_info.height - config[1] #FLIP
        pose = our_map_to_world(config, map_info)
        config[0] = pose[0]
        config[1] = pose[1]
      configs.append(config)
  
  return configs

# PriorityQueue sourced from UCB AI Pacman assignments
class PriorityQueue:

    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

if __name__ == '__main__':
  print world_to_map([64,64,20])
