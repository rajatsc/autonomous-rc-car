#!/usr/bin/env python

import rospy 
import numpy as np
from nav_msgs.srv import GetMap
from lab4.srv import *
from HaltonPlanner import HaltonPlanner
from HaltonEnvironment import HaltonEnvironment
import KinematicModel as model

PLANNER_SERVICE_TOPIC = '/planner_node/get_plan' # The topic that the planning service is provided on
POST_PROCESS_MAX_TIME = 5.0

class PlannerNode(object):

  def __init__(self):
    # Get the map    
    map_service_name = rospy.get_param("~static_map", "/mppi/static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    self.map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
        
    graph_file = rospy.get_param("~graph_file", '../grap_files/cse_floor_four_sparse.graphml') # Get the name of the halton graph file
    print("graph file", graph_file)
    self.visualize = rospy.get_param("~visualize", True)
    self.environment = HaltonEnvironment(self.map_msg, graph_file, None, None) # Create the Halton Environment
    self.planner = HaltonPlanner(self.environment) # Create the Halton Planner
    self.server = rospy.Service(PLANNER_SERVICE_TOPIC, GetPlan, self.plan_cb) # Offer planning service
    print 'Ready to plan'
    
  # Called when client wants to generate a plan  
  def plan_cb(self, req):
    # Check that target and source have correct dimension
    if len(req.source) != model.SPACE_DIM or len(req.target) != model.SPACE_DIM:
      return [[],False]
    
    # Check for source == target
    if req.source == req.target:
      result = []
      result.extend(req.source)
      result.extend(req.target)
      return [result, True]
      
    source = np.array(req.source).reshape(3)
    target = np.array(req.target).reshape(3)

    self.environment.set_source_and_target(source, target) # Tell environment what source and target are
    
    # Check if planning is trivially infeasible on this environment
    if not self.environment.manager.get_state_validity(source):
      print 'Source in collision'
      return [[],False]

    if not self.environment.manager.get_state_validity(target):
      print 'Target in collision'
      return [[],False]
      
    plan = self.planner.plan() # Find a plan
    
    if plan:
      self.planner.simulate(plan, "simulation_pre.png") # Visualize plan
      plan = self.planner.post_process(plan, POST_PROCESS_MAX_TIME) # Try to improve plan
      if self.visualize:
        print "gonna viz now!"
        self.planner.simulate(plan, "simulation_post.png") # Visualize plan
      flat_plan = [el for config in plan for el in config] # Convert to a flat list
      return [flat_plan, True]
    else:
      return [[],False]
    
if __name__ == '__main__':
  rospy.init_node('planner_node', anonymous=True)

  pn = PlannerNode()
  
  rospy.spin()  
