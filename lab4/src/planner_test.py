#!/usr/bin/env python

import rospy 
import numpy as np
from lab4.srv import *

PLANNER_SERVICE_TOPIC = '/planner_node/get_plan' # The topic at which the service is available

SOURCE = [-48.68, -0.28, 0]   # Where the plan should start
TARGET = [1.56,    -34.16,    0] # Where the plan should finish

#SOURCE = [0.28,  8.44, 0.0]
#TARGET = [-44.92,  -18.68,1.57]

#SOURCE = [-8.76,  -2.04, 0.0]
#TARGET = [-10.68,  -24.28, 0.0]

if __name__ == '__main__':

  rospy.init_node('planner_test', anonymous=True) # Initialize the node
  rospy.wait_for_service(PLANNER_SERVICE_TOPIC) # Wait for the service to become available if necessary
  get_plan = rospy.ServiceProxy(PLANNER_SERVICE_TOPIC, GetPlan) # Setup a ros service client
  
  try:
    resp = get_plan(SOURCE,TARGET) # Get the plan from the service
    print np.array(resp.plan).reshape(-1,3) # Reshape the plan to be nx3
    print resp.success
  except rospy.ServiceException, e:
    print 'Service call failed: %s'%e
    

