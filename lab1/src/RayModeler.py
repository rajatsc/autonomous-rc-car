#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import rospy
import range_libc
import time
import rosbag
from threading import Lock

from nav_msgs.srv import GetMap

if __name__ == '__main__':
  rospy.init_node('ray_modeler', anonymous=True)
  bag_path = rospy.get_param("laser_scan_bag_local")
  scan_msg = list(rosbag.Bag(bag_path).read_messages())[0][1]
  angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
  angle_rays = zip(angles, scan_msg.ranges)
  x_coords = [ r * np.cos(theta) for theta, r in angle_rays ]  
  y_coords = [ r * np.sin(theta) for theta, r in angle_rays ]  
  np.savetxt('/home/nvidia/corner.txt', np.array(zip(x_coords,y_coords)))
  plt.scatter(x_coords, y_coords)
  plt.show()
