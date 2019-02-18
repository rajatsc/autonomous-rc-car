#!/usr/bin/env python
from threading import Lock
from lab4.msg import *
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Float64
import numpy as np
import rospy
import Utils

class PathPublisher(object):
  XY_THRESHOLD = 1
  THETA_THRESHOLD = np.pi # TODO decide if we want theta threshold
  XY_GOAL_THRESHOLD = 0.35
  THETA_GOAL_THRESHOLD = np.pi # TODO decide if we want theta threshold
  XY_OBS_THRESHOLD = 1
  OBSTACLE_VEL = 0.95
  MAX_VEL = 1.5
  
  def __init__(self):
    pub_topic_goal = "/pp/path_goal"
    pub_topic_max_vel = "/pp/max_vel"
    sub_topic_path = "/multi_planner/mppi_path"
    sub_topic_cur_loc = "/pf/ta/viz/inferred_pose"

    self.state_lock = Lock()
    self.cur_path_idx = 0 # Index of current path in self.paths
    self.cur_dest_idx = 0 # Index of current destination in self.paths[self.cur_path_idx]
    self.paths = [] # List of paths to process. First element is source, last element is source
    self.non_permissible_region = np.load('/home/nvidia/catkin_ws/src/lab4/maps/permissible_region.npy')[::-1,:]

    map_service_name = rospy.get_param("~static_map", "/planning/static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    self.map_info = rospy.ServiceProxy(map_service_name, GetMap)().map.info

        
    bad_waypoints_csv = rospy.get_param("~bad_waypoints_csv", "/home/nvidia/catkin_ws/src/lab4/final/bad_waypoints.csv")
    mode = rospy.get_param("~mode", "pixel")
    self.obstacles = map(np.array, Utils.load_csv_to_configs(bad_waypoints_csv, mode, self.map_info))

    self.goal_pub = rospy.Publisher(pub_topic_goal, PoseStamped, queue_size=10)
    self.max_vel_pub = rospy.Publisher(pub_topic_max_vel, Float64, queue_size=10)
    path_sub = rospy.Subscriber(sub_topic_path, MPPIPath, self.path_cb)
    cur_loc_sub = rospy.Subscriber(sub_topic_cur_loc, PoseStamped, self.location_cb)
    print "Ready to receive path!"


  def process_mppi_path(self, msg):
    path = msg.path
    paths = []
    for pose_arr in path:
      sub_path = []
      for pose in pose_arr.poses:
        sub_path.append(np.array(Utils.pose_to_config(pose)))
      paths.append(sub_path)
    return paths
      
  def path_cb(self, msg):
    rospy.logerr("Received path!")
    self.state_lock.acquire()
    self.paths = self.process_mppi_path(msg)
    self.cur_path_idx = 0
    self.cur_dest_idx = 0 
    self.state_lock.release()
    goal = self.get_next_dest()
    rospy.logerr("Publishing new goal")
    self.goal_pub.publish(Utils.config_to_posestamped(goal))

  def location_cb(self, msg):
    # gets current location
    # checks to see 
    curr_pose = np.array(Utils.posestamped_to_config(msg))
    if len(self.paths) == 0:
      return
    if self.near_cur_dest(curr_pose):
      goal = self.get_next_dest()
      if goal == None:
        return
      rospy.logerr("Publishing new goal")
      self.goal_pub.publish(Utils.config_to_posestamped(goal))
      
    if self.near_obstacle(curr_pose):
      self.max_vel_pub.publish(Float64(self.OBSTACLE_VEL))
      print "near obstacle"
    else:
      self.max_vel_pub.publish(Float64(self.MAX_VEL))
  
  def near_cur_dest(self, curr_pose):
    if len(self.paths) == 0:
      return False
    dest = self.paths[self.cur_path_idx][self.cur_dest_idx]
    difference_from_dest = np.abs(curr_pose - dest)
    xy_distance_to_dest = np.linalg.norm(difference_from_dest[:2])
    theta_distance_to_dest = difference_from_dest[2] % (2 * np.pi)
    if self.dest_is_goal():
      return xy_distance_to_dest < self.XY_GOAL_THRESHOLD and theta_distance_to_dest < self.THETA_GOAL_THRESHOLD
    else:
      return xy_distance_to_dest < self.XY_THRESHOLD# and theta_distance_to_dest < self.THETA_THRESHOLD

  def near_obstacle(self, curr_pose):
    if len(self.paths) == 0:
      return False
    for obstacle in self.obstacles:
      difference_from_obs = np.abs(curr_pose - obstacle)
      xy_distance_to_obs = np.linalg.norm(difference_from_obs[:2])
      if xy_distance_to_obs < self.XY_OBS_THRESHOLD:
        return True
    return False

  def get_next_dest(self):
    self.state_lock.acquire()
    while True:
      # Try to advance dest_idx within path
      if self.cur_dest_idx < len(self.paths[self.cur_path_idx])-1:
        self.cur_dest_idx += 1
        if self.dest_is_goal(): 
          self.state_lock.release()
          return self.paths[self.cur_path_idx][self.cur_dest_idx]
      # Otherwise advance to beginning of next path (if possible)
      elif self.cur_path_idx < len(self.paths)-1:
        self.cur_path_idx += 1
        self.cur_dest_idx = 0
        if self.dest_is_goal(): 
          self.state_lock.release()
          return self.paths[self.cur_path_idx][self.cur_dest_idx]
      else:
        rospy.logerr("Route completed! No more paths to publish!")
        self.state_lock.release()
        return None
      config = self.paths[self.cur_path_idx][self.cur_dest_idx]
      map_config = Utils.our_world_to_map(config, self.map_info)
      if self.non_permissible_region[int(map_config[1]),int(map_config[0])]:
        continue
      else:
        self.state_lock.release()
        return config

  def dest_is_goal(self):
    return self.cur_dest_idx == len(self.paths[self.cur_path_idx]) - 1

if __name__ == "__main__":
  rospy.init_node("path_publisher", anonymous=True)
  
  pp = PathPublisher()
  rospy.spin()
