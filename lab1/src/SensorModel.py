#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rospy
import range_libc
import time
import rosbag
from threading import Lock
import utils

from nav_msgs.srv import GetMap

THETA_DISCRETIZATION = 112 # Discretization of scanning angle
INV_SQUASH_FACTOR = 0.2    # Factor for helping the weight distribution to be less peaked

Z_SHORT = 0.01 #0.14  # Weight for short reading
LAMBDA_SHORT = 0.05 # Intrinsic parameter of the exponential dist.
Z_MAX = 0.07 #0.015    # Weight for max reading
Z_RAND = 0.12 #0.045 # Weight for random reading
SIGMA_HIT = 8.0 # Noise value for hit reading
Z_HIT = 0.80 #0.80  # Weight for hit reading

HEATMAP_NUM_THETA = 70  # num thetas to iterate for the heat map
HEATMAP_XY_STEP = 1

class SensorModel:
	
  def __init__(self, map_msg, particles, weights, state_lock=None):
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
  
    self.particles = particles
    self.weights = weights
    
    self.LASER_RAY_STEP = int(rospy.get_param("~laser_ray_step")) # Step for downsampling laser scans
    self.MAX_RANGE_METERS = float(rospy.get_param("~max_range_meters")) # The max range of the laser
    
    oMap = range_libc.PyOMap(map_msg) # A version of the map that range_libc can understand
    max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution) # The max range in pixels of the laser
    self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION) # The range method that will be used for ray casting
    self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px)) # Load the sensor model expressed as a table

    self.queries = None
    self.ranges = None
    self.laser_angles = None # The angles of each ray
    self.downsampled_angles = None # The angles of the downsampled rays 
    self.do_resample = False # Set so that outside code can know that it's time to resample
    
  def lidar_cb(self, msg):
    self.state_lock.acquire()

    # Compute the observation
    # obs is a a two element tuple
    # obs[0] is the downsampled ranges
    # obs[1] is the downsampled angles
    # Each element of obs must be a numpy array of type np.float32 (this is a requirement for GPU processing)
    # Use self.LASER_RAY_STEP as the downsampling step
    # Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
  
    # YOUR CODE HERE
    if self.downsampled_angles is None:   
      self.downsampled_angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment * self.LASER_RAY_STEP, dtype=np.float32)
      self.sample_indices = np.arange(0, self.downsampled_angles.shape[0] * self.LASER_RAY_STEP, self.LASER_RAY_STEP, dtype=np.int)

    ranges = np.array(msg.ranges, dtype=np.float32)
    downsampled_ranges = ranges[self.sample_indices]
    obs = (downsampled_ranges, self.downsampled_angles)

    self.apply_sensor_model(self.particles, obs, self.weights)
    self.weights /= np.sum(self.weights)
    
    self.last_laser = msg
    self.do_resample = True

    self.state_lock.release()

  def normal(self, mean, sd, x):
    return np.exp(-(1.0/2)*((x-mean)**2)/(1.0*sd**2))/(np.sqrt(2*np.pi*(sd**2)))

  def precompute_sensor_model(self, max_range_px):
    rospy.logerr("started precompute sensor model")

    table_width = int(max_range_px) + 1
    sensor_model_table = np.zeros((table_width,table_width))
    # Populate sensor model table as specified
    for ztk_star in range(table_width):
      hit_norm = 0 
      for ztk in range(table_width): # P_hit
        sensor_model_table[ztk,ztk_star] = self.normal(ztk_star, SIGMA_HIT, ztk)  
        hit_norm += sensor_model_table[ztk,ztk_star]
      sensor_model_table[:,ztk_star] *= Z_HIT / hit_norm
      if ztk_star != 0:
        short_norm = 1 / (1 - np.exp(-LAMBDA_SHORT * ztk_star))
      else:
	      short_norm = 0
      for ztk in range(table_width):
        if ztk <= ztk_star:
          sensor_model_table[ztk,ztk_star] += short_norm * LAMBDA_SHORT * np.exp(-LAMBDA_SHORT * ztk) * Z_SHORT # P_short
        sensor_model_table[ztk,ztk_star] += Z_RAND / max_range_px #P_rand
        if ztk == max_range_px: 
          sensor_model_table[ztk,ztk_star] += Z_MAX # P_max
      sensor_model_table[:,ztk_star] /= np.sum(sensor_model_table[:,ztk_star])

    rospy.logerr("done precomputing sensor model")
    # plot precomputed distribution
    #spot_checks = [0, 20, 150, 200, 280]
    #for col in spot_checks:
    #  plt.scatter(np.arange(0,table_width), sensor_model_table[:,col])
    #  plt.show()

    return sensor_model_table

  def apply_sensor_model(self, proposal_dist, obs, weights):

    obs_ranges = obs[0]
    obs_angles = obs[1]
    num_rays = obs_angles.shape[0]
    
    # Only allocate buffers once to avoid slowness
    if not isinstance(self.queries, np.ndarray):
      self.queries = np.zeros((proposal_dist.shape[0],3), dtype=np.float32)
      self.ranges = np.zeros(num_rays*proposal_dist.shape[0], dtype=np.float32)

    self.queries[:,:] = proposal_dist[:,:]

    self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    # Evaluate the sensor model on the GPU
    self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

    np.power(weights, INV_SQUASH_FACTOR, weights)

def generate_heatmap(map_resp, scan_msg):
  width = map_resp.map.info.width
  height = map_resp.map.info.height
  heatmap = np.zeros((height, width))
  map_data = np.array(map_resp.map.data).reshape(heatmap.shape)
  
  # particles (in map pixels) where map is permissible
  pixel_particles = np.zeros((width*height, 3))
  # populate all indices
  pixel_particles[:,0] = np.tile(np.arange(width), height)
  pixel_particles[:,1] = np.repeat(np.arange(height), width)
  # filter particles based on map
  pixel_particles = pixel_particles.reshape(height, width, 3)[map_data >= 0]

  # downsample particles
  downsampled_indices = np.arange(0, pixel_particles.shape[0], HEATMAP_XY_STEP)
  pixel_particles = pixel_particles[downsampled_indices]
  
  # transform particles into "world" frame (in meters, etc)
  meter_particles = np.copy(pixel_particles)
  utils.map_to_world(meter_particles, map_resp.map.info)

  num_particles = meter_particles.shape[0]
  weights = np.ones(num_particles) / float(num_particles)

  sensor_model = SensorModel(map_resp.map, meter_particles, weights)

  max_weights = np.zeros(num_particles)
  # iterate through all thetas
  for theta in np.linspace(-np.pi, np.pi, HEATMAP_NUM_THETA):
    meter_particles[:,2] = theta
    sensor_model.lidar_cb(scan_msg)
    # update the max weights
    np.maximum.reduce((weights, max_weights), out=max_weights)

  max_weights /= np.sum(max_weights)
  # populate heatmap according to max_weights
  for i in range(num_particles):
    px, py = pixel_particles[i,0:2]
    heatmap[int(py),int(px)] = max_weights[i]

  rospy.logerr("done computing heatmap")

  return heatmap

def plot_heatmap(map_data, heatmap):
  plt.imsave("/home/nvidia/heatmap.png", heatmap, cmap="Reds")
  plt.imsave("/home/nvidia/map.png", map_data, cmap="hot")
  plt.figure()

  # draw map background
  plt.imshow(map_data, cmap='Greens')
  
  # set values < 0.05 to alpha=0
  # overlay heatmap onto background
  cmap = cm.Reds
  cmap.set_under('k', alpha=0)
  num_particles = np.count_nonzero(map_data)
  upper = np.percentile(heatmap, 99.997) #2.0/num_particles # np.max(heatmap)
  lower = max(np.percentile(heatmap, 98), upper/100.0) #1.0/num_particles
  plt.imshow(heatmap, cmap=cmap, clim=[lower, upper])

  plt.savefig("/home/nvidia/heatmap_overlay.png", dpi=500)

  rospy.logerr("heatmap_overaly.png saved!")

if __name__ == '__main__':
  rospy.init_node('sensor_model', anonymous=True)
  
  # get map
  rospy.wait_for_service('static_map')
  static_map = rospy.ServiceProxy('/static_map', GetMap)
  rospy.logerr("Done waiting for map service.")
  map_resp = static_map()
  map_data_shaped = np.array(map_resp.map.data).reshape((map_resp.map.info.height,map_resp.map.info.width))
  map_data_shaped = map_data_shaped >= 0

  # read laser scan from bag file
  bag_path = rospy.get_param("laser_scan_bag")
  scan_msg = list(rosbag.Bag(bag_path).read_messages())[0][1]

  # generate and plot heatmap
  heatmap = generate_heatmap(map_resp, scan_msg)
  plot_heatmap(map_data_shaped, heatmap)

  # plot discretization vs runtime
  #for theta in [3,5,7,10,15,20,50]:
  #  HEATMAP_NUM_THETA = theta
  #for step in [200, 100, 50, 20, 10, 5, 3, 1]:
  #  HEATMAP_XY_STEP = step
  #  start = time.time()
  #  heatmap = generate_heatmap(map_resp, scan_msg)
  #  plot_heatmap(map_data_shaped, heatmap)
  #  end = time.time()
    
  #  rospy.logerr("step:{}; time:{}".format(step, (end - start)))  
  #  rospy.logerr("theta:{}; time:{}".format(theta, (end - start)))

