#!/usr/bin/env python

import rospy
import csv
import sys
import numpy as np
import math
import Utils

from nav_msgs.srv import GetMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty
from lab4.srv import *
from lab4.msg import *

PLANNER_SERVICE_TOPIC = '/planner_node/get_plan' # The topic at which the service is available

class MultiPlanner(object):

  MARKER_SIZE = 0.3
  GET_SERVER = True 
  GRAND_PLAN_FILE = 'grand_plan.npy'

  def __init__(self, blue_configs, start_config=None, csv_start_config=None, order_mode="euclidian", grand_plan=None, visualize=True):
    self.VISUALIZE = visualize
    self.ORDER_MODE = order_mode

    self.blue_configs = blue_configs 
    self.start_config = start_config
    self.csv_start_config = csv_start_config
    self.grand_plan = grand_plan
    self.cur_config = None

    if self.GET_SERVER:
      rospy.loginfo("Waiting for planner service...")
      rospy.wait_for_service(PLANNER_SERVICE_TOPIC) # Wait for the service to become available if necessary
      rospy.loginfo("Got planner service")
      self.planner_client = rospy.ServiceProxy(PLANNER_SERVICE_TOPIC, GetPlan) # Setup a ros service client

    # visualization publishers
    self.viz_waypoint_pub = rospy.Publisher('/multi_planner/waypoints', Marker, queue_size=10)
    self.viz_plan_pub = rospy.Publisher('/multi_planner/plan_path', Path, queue_size=10)
    self.viz_plan_nodes_pub = rospy.Publisher('/multi_planner/plan_nodes', PoseArray, queue_size=10)
    self.viz_target_pub = rospy.Publisher('/multi_planner/target', PoseStamped, queue_size=10)

    self.mppi_pub = rospy.Publisher('/multi_planner/mppi_path', MPPIPath, queue_size=10)

    self.pose_sub = rospy.Subscriber("/pf/ta/viz/inferred_pose", PoseStamped, self.pose_cb, queue_size=1)
    self.joy_sub = rospy.Subscriber("/vesc/joy", Joy, self.joy_cb, queue_size=1)

    # Programmatic control subscribers (used with ExperimentTool)
    self.execute_sub = rospy.Subscriber("/multi_planner/execute", Empty, self.execute_cb, queue_size=1)
    self.plan_from_pose_sub = rospy.Subscriber("/multi_planner/plan_from_pose", PoseStamped, 
                                  self.plan_from_pose_cb, queue_size=1)
    self.planning_confirm_pub = rospy.Publisher("/exp_tool/planning_confirm", Empty, queue_size=1)
    
    rospy.loginfo("MultiPlanner done init")

  def plan_from_pose_cb(self, msg):
    start = Utils.posestamped_to_config(msg)
    rospy.loginfo("START requested!  " + str(start) + " start config: " + str(self.start_config))
    rospy.loginfo("Start config: {}".format(start))
    if (self.start_config is not None and np.array_equal(start[:2], self.start_config[:2])):
      
      self.planning_confirm_pub.publish(Empty())
      return
    self.start_config = start 

    self.start_planning()
    self.planning_confirm_pub.publish(Empty())

  def execute_cb(self, msg):
    if self.grand_plan:
      rospy.loginfo("EXECUTE requested!")
      self.execute_plan()
    else:
      rospy.logerr("EXECUTE requested, but no grand_plan yet!!")

  def pose_cb(self, msg):
    self.cur_config = Utils.posestamped_to_config(msg)

  def joy_cb(self, msg):
    # buttons=[A, B, X, Y, LB, RB, Back, Start, Logitech, Left joy, Right joy]

    # Y is pressed, get current pose as starting pose
    # and start planning
    if msg.buttons[3]:
      if self.cur_config:
        self.start_config = self.cur_config
        rospy.loginfo("START requested!")
        rospy.loginfo("Start config: {}".format(self.start_config))

        self.start_planning()
        
      else:
        rospy.logerr("START requested, but no cur_pose yet!!")

    # B is pressed, plan from csv start config
    if msg.buttons[1]:
      if self.csv_start_config:
        self.start_config = self.csv_start_config
        rospy.loginfo("Start planning from CSV start config: {}".format(self.start_config))
        
        self.start_planning()
      else:
        rospy.logerr("No CSV start config given!")

    # A is pressed, start executing the plan
    if msg.buttons[0]:
      self.execute_cb(Empty())

  def euclidian_distance(self, config1, config2):
    return math.sqrt((config1[0] - config2[0])**2 + (config1[1] - config2[1])**2)

  def euclidian_order(self):
    configs = self.blue_configs[:]
    path = [self.start_config]

    # first, get the path order according to euclidian dist
    while len(configs) != 0:
      cur_source = path[-1]
      min_dist = self.euclidian_distance(cur_source, configs[0])
      min_config = configs[0]
      
      for cur_config in configs:    
        dist = self.euclidian_distance(cur_source, cur_config)
        if dist < min_dist:
          min_dist = dist
          min_config = cur_config 
      
      configs.remove(min_config)
      path.append(min_config)

    return path

  def straight_line_theta(self, config1, config2):
    dx = config2[0] - config1[0]
    dy = config2[1] - config1[1]
    theta = np.arctan2(dy, dx)
    return (theta + 2*np.pi) % (2*np.pi)

  def populate_theta(self, path):
    # second, get the best guess theta for the waypoints
    for i in range(0, len(path)-1):
      cur_config = path[i]
      next_config = path[i+1]
      theta = self.straight_line_theta(cur_config, next_config)
      
      # add theta to coord
      cur_config[2] = theta
      path[i] = cur_config

      # if next_coord is the last element
      if (i + 1) == (len(path) - 1):
        next_config[2] = theta
        path[i + 1] = next_config


  def start_planning(self):
    rospy.loginfo("Start planning...")

    if self.ORDER_MODE == "euclidian":
      path = self.euclidian_order()  # [(x,y,th),(x,y,th),...]
    else: # manual
      path = [self.start_config]
      path.extend(self.blue_configs)

    self.populate_theta(path)

    rospy.loginfo("Waypoints order: {}".format(path))

    # [np.array(n,3), np.array(n,3), ...]
    grand_plan = []

    for i in range(len(path) - 1):
      source = path[i]  # [x,y,th]
      target = path[i + 1]

      if self.VISUALIZE:
        self.visualize_target(target)

      rospy.loginfo("Getting plan for {} to {}..".format(source, target))
      # send request to planner node
      try:
        resp = self.planner_client(source, target) # Get the plan from the service
        plan = np.array(resp.plan).reshape(-1,3) # Reshape the plan to be (nx3,) -> (n,3)
        
        if plan.shape[0] == 0:
          rospy.logerr("NO VALID PLAN!! ALL DEPENDS ON MPPI!! REALLY CONSIDER ADD POINTS AND REPLAN!!")
          plan = np.array([source, target]) # (2,3)

        grand_plan.append(plan)
        print plan
        print resp.success

        if self.VISUALIZE:
          self.visualize_plan_nodes(grand_plan)
          self.visualize_plan(grand_plan)

      except rospy.ServiceException, e:
        print 'Service call failed: %s'%e

    self.grand_plan = grand_plan

    rospy.loginfo("Saving grand plan..")
    np.save(self.GRAND_PLAN_FILE, self.grand_plan)
    rospy.loginfo("Saved to {}".format(self.GRAND_PLAN_FILE))

    rospy.loginfo("FINISHED PLANNING!!")

  def execute_plan(self):
    rospy.loginfo("Start executing the grand plan...")
    
    # TODO pre-process points for MPPI
    execution_plan = self.pre_process_grand_plan()

    # convert grand_plan into MPPI msg
    mppi_path = MPPIPath()
    mppi_path.header.frame_id = "map"
    for path in execution_plan:
      posearray = Utils.plan_to_posearray(path)
      mppi_path.path.append(posearray)

    # publish
    self.mppi_pub.publish(mppi_path)

    rospy.loginfo("DONE EXECUTING!!")


  def pre_process_grand_plan(self):
    PATH_FILTER_SIZE = rospy.get_param("path_filter_size", 5)

    execution_plan = []
    for path in self.grand_plan:
      # path: np.array(nx3)
      filter_indices = range(0, path.shape[0], PATH_FILTER_SIZE)

      # make sure we have the final goal in the filered path
      if path.shape[0] > 0 and filter_indices[-1] != (path.shape[0] - 1):
        filter_indices.append(path.shape[0] - 1)

      execution_plan.append(path[filter_indices])

    return execution_plan


  ############ Visualizations #####################

  def visualize_waypoints(self):
    if self.start_config:
      marker = self.make_marker(self.start_config, 0, "start")
      self.viz_waypoint_pub.publish(marker)

    i = 1
    for config in self.blue_configs:
      marker = self.make_marker(config, i, "waypoint")
      self.viz_waypoint_pub.publish(marker)
      i += 1

  def make_marker(self, config, i, point_type):
    marker = Marker()
    marker.header = Utils.make_header('map')
    marker.ns = str(config)
    marker.id = i
    marker.type = Marker.CUBE
    marker.pose.position.x = config[0]
    marker.pose.position.y = config[1]
    marker.pose.orientation.w = 1
    marker.scale.x = self.MARKER_SIZE
    marker.scale.y = self.MARKER_SIZE
    marker.scale.z = self.MARKER_SIZE
    marker.color.a = 1.0
    if point_type == "waypoint":
      marker.color.b = 1.0
    else:
      marker.color.g = 1.0

    return marker

  def visualize_target(self, target_config):
    target_pose = Utils.config_to_posestamped(target_config)
    self.viz_target_pub.publish(target_pose)

  def visualize_plan(self, grand_plan):
    # grand_plan [array, array, ..]
    path = Path()
    path.header = Utils.make_header('map')
    # plan: N x 3
    # grand_plan: [ Nx3, Mx3, Kx3, ...] -> (N+M+K)x3
    path.poses = map(Utils.config_to_posestamped, np.vstack(grand_plan))
    self.viz_plan_pub.publish(path)

  def visualize_plan_nodes(self, grand_plan):
    # plan: N x 3
    # grand_plan: [ Nx3, Mx3, Kx3, ...] -> (N+M+K)x3
    nodes = Utils.plan_to_posearray(np.vstack(grand_plan))
    self.viz_plan_nodes_pub.publish(nodes)

  ####################################################

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
        pose = Utils.our_map_to_world(config, map_info)
        config[0] = pose[0]
        config[1] = pose[1]
      configs.append(config)
  
  return configs


if __name__ == '__main__':
  
  rospy.init_node('multi_planner', anonymous=True) # Initialize the node
  Utils.wait_for_time()

  # Get the map info
  map_service_name = rospy.get_param("~static_map", "static_map")
  print("Getting map from service: ", map_service_name)
  rospy.wait_for_service(map_service_name)
  print("Done getting map service")
  map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
  map_info = map_msg.info

  # Params
  csv_file = rospy.get_param("~csv_file")
  csv_mode = rospy.get_param("~csv_mode")
  order_mode = rospy.get_param("~order_mode")
  visualize = rospy.get_param("~visualize")
  testing = rospy.get_param("~testing") 
  start_csv_file = rospy.get_param("~start_csv_file", None)
  grand_plan_file = rospy.get_param("~grand_plan_file", None)

  # Load blue wayponts from csv
  blue_configs = load_csv_to_configs(csv_file, csv_mode, map_info)

  # Load start config from csv if given
  start_config = None
  if testing:
    if start_csv_file:
      start_config = load_csv_to_configs(start_csv_file, csv_mode, map_info)[0]
    else:
      start_config = [0.0, 0.0, 0.0]

  csv_start_config = None
  if start_csv_file:
    csv_start_config = load_csv_to_configs(start_csv_file, csv_mode, map_info)[0]

  # load grand plan from file if given
  grand_plan = None
  if grand_plan_file:
    grand_plan = list(np.load(grand_plan_file))

  mp = MultiPlanner(blue_configs, 
                    start_config=start_config,
                    csv_start_config=csv_start_config,
                    order_mode=order_mode, 
                    grand_plan=grand_plan, 
                    visualize=visualize)
  rospy.sleep(1)

  if testing:
    if visualize:
      mp.visualize_waypoints()
    mp.start_planning()
    mp.execute_plan()
  else:
    if visualize:
      while True:
        mp.visualize_waypoints()
    else:
      rospy.spin()
  

