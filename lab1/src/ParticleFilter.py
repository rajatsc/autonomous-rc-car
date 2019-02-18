#!/usr/bin/env python

import matplotlib.pyplot as plt

import rospy 
import numpy as np
import time
import utils 
import tf.transformations
import tf
from threading import Lock

from vesc_msgs.msg import VescStateStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from ReSample import ReSampler
from SensorModel import SensorModel
from MotionModel import OdometryMotionModel, KinematicMotionModel

 
class ParticleFilter():

  def __init__(self):
    self.MAX_PARTICLES = int(rospy.get_param("~max_particles")) # The maximum number of particles
    self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles")) # The maximum number of particles to visualize
    self.noise = bool(rospy.get_param("~noise"))

    self.particle_indices = np.arange(self.MAX_PARTICLES)
    self.particles = np.zeros((self.MAX_PARTICLES,3)) # Numpy matrix of dimension MAX_PARTICLES x 3
    self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES) # Numpy matrix containig weight for each particle
    self.viz_seq = 0
    self.INIT_NOISE_POS_SIGMA = 0.2
    self.INIT_NOISE_THETA_SIGMA = np.pi/12

    self.state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this
    
    # Use the 'static_map' service (launched by MapServer.launch) to get the map
    # Will be used to initialize particles and SensorModel
    # Store map in variable called 'map_msg'
    # YOUR CODE HERE
    rospy.wait_for_service('static_map')
    static_map = rospy.ServiceProxy('/static_map', GetMap)
    map_msg = static_map().map

    # Globally initialize the particles
    self.initialize_global(map_msg)
   
    # Publish particle filter state
    self.pose_pub      = rospy.Publisher("/pf/viz/inferred_pose", PoseStamped, queue_size = 1) # Publishes the expected pose
    self.particle_pub  = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1) # Publishes a subsample of the particles
    self.pub_tf = tf.TransformBroadcaster() # Used to create a tf between the map and the laser for visualization
    self.pub_laser     = rospy.Publisher("/pf/viz/scan", LaserScan, queue_size = 1) # Publishes the most recent laser scan

    self.RESAMPLE_TYPE = rospy.get_param("~resample_type", "naiive") # Whether to use naiive or low variance sampling
    self.resampler = ReSampler(self.particles, self.weights, self.state_lock)  # An object used for resampling

    self.sensor_model = SensorModel(map_msg, self.particles, self.weights, self.state_lock) # An object used for applying sensor model
    self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.sensor_model.lidar_cb, queue_size=1)
    
    self.MOTION_MODEL_TYPE = rospy.get_param("~motion_model", "kinematic") # Whether to use the odometry or kinematics based motion model
    if self.MOTION_MODEL_TYPE == "kinematic":
      self.motion_model = KinematicMotionModel(self.particles, self.state_lock, noise=self.noise) # An object used for applying kinematic motion model
      self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/sensors/core"), VescStateStamped, self.motion_model.motion_cb, queue_size=1)
    elif self.MOTION_MODEL_TYPE == "odometry":
      self.motion_model = OdometryMotionModel(self.particles, self.state_lock, noise=self.noise)# An object used for applying odometry motion model
      self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/odom"), Odometry, self.motion_model.motion_cb, queue_size=1)
    else:
      print "Unrecognized motion model: "+ self.MOTION_MODEL_TYPE
      assert(False)
    
    # Use to initialize through rviz. Check clicked_pose_cb for more info    
    self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)

  # Initialize the particles to cover the map
  def initialize_global(self, map_msg):
    # YOUR CODE HERE
    pass
    
  # Publish a tf between the laser and the map
  # This is necessary in order to visualize the laser scan within the map
  def publish_tf(self,pose):
  # Use self.pub_tf
  # YOUR CODE HERE
    orientation = tf.transformations.quaternion_from_euler(0,0,pose[2])

    cur_time = rospy.Time.now()
    self.pub_tf.sendTransform((pose[0], pose[1], 0),  # translation
                              orientation,            # rotation in quaternion
                              cur_time,       # time
                              "laser",                # child frame_id
                              "map")                  # parent frame_id

    return cur_time

  # Returns the expected pose given the current particles and weights
  def expected_pose(self):
    num_particles = self.particles.shape[0]
    # [x_avg, y_avg, theta_avg] (weighted avg)
    return np.sum(self.particles * self.weights.repeat(3).reshape(num_particles,3), axis=0)
    
    
  # Callback for '/initialpose' topic. RVIZ publishes a message to this topic when you specify an initial pose using its GUI
  # Reinitialize your particles and weights according to the received initial pose
  # Remember to apply a reasonable amount of Gaussian noise to each particle's pose
  def clicked_pose_cb(self, msg):
    rospy.logerr("consulted!")
    self.state_lock.acquire()
    
    # YOUR CODE HERE
    pos = msg.pose.pose.position
    quat = msg.pose.pose.orientation
    yaw = utils.quaternion_to_angle(quat)

    self.particles[:,0] = pos.x + self.INIT_NOISE_POS_SIGMA*np.random.randn(self.particles.shape[0])
    self.particles[:,1] = pos.y + self.INIT_NOISE_POS_SIGMA*np.random.randn(self.particles.shape[0])
    self.particles[:,2] = yaw + self.INIT_NOISE_THETA_SIGMA*np.random.randn(self.particles.shape[0])

    self.state_lock.release()
    
  # Visualize the current state of the filter
  # (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
  # (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be the child_frame_id of the tf from (1)
  # (3) Publishes a PoseStamped message indicating the expected pose of the car
  # (4) Publishes a subsample of the particles (use self.MAX_VIZ_PARTICLES). 
  #     Sample so that particles with higher weights are more likely to be sampled.
  def visualize(self):
    self.state_lock.acquire()
    # (1) Publish tf between map and the laser. Depends on the expected pose.
    pose = self.expected_pose()
    tf_time = self.publish_tf(pose)
    # (2) Publish most recent laser measurement
    msg = self.sensor_model.last_laser
    msg.header.frame_id = "laser"
    msg.header.stamp = tf_time

    self.pub_laser.publish(msg) 
    # (3) Publish PoseStamped message indicating the expected pose of the car.  
    pose_msg = PoseStamped()
    pose_msg.header.seq = self.viz_seq
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "map"
    pose_msg.pose = utils.point_to_pose(pose)
    self.pose_pub.publish(pose_msg) 

    # (4) Publish a subsample of the particles 
    particle_indices = np.random.choice(np.arange(self.particles.shape[0]), size=self.MAX_VIZ_PARTICLES, replace=True, p=self.weights)
    particle_sample = self.particles[particle_indices]
    particles_msg = PoseArray()
    particles_msg.header.seq = self.viz_seq
    particles_msg.header.stamp = rospy.Time.now()
    particles_msg.header.frame_id = "map"
    particles_msg.poses = [utils.point_to_pose(p) for p in particle_sample]
    self.particle_pub.publish(particles_msg)

    #rospy.logerr(particles_msg)

    self.viz_seq += 1 
    self.state_lock.release()
  
# Suggested main 
if __name__ == '__main__':
  rospy.init_node("particle_filter", anonymous=True) # Initialize the node
  pf = ParticleFilter() # Create the particle filter
 
  pf.publish_tf(pf.expected_pose())

  # plot computation time vs num particles
  i = 0
  times = []
  def plot_time():
    xs, ys = zip(*times)
    plt.scatter(xs, ys)
    plt.show()
    print 'Xs: iter num'
    print xs
    print '\nYs: computation time'
    print ys
  rospy.on_shutdown(plot_time)
 
  while not rospy.is_shutdown(): # Keep going until we kill it
    # Callbacks are running in separate threads
    if pf.sensor_model.do_resample: # Check if the sensor model says it's time to resample
      pf.sensor_model.do_resample = False # Reset so that we don't keep resampling

      start = time.time()

      # Resample
      if pf.RESAMPLE_TYPE == "naiive":
        pf.resampler.resample_naiive()
      elif pf.RESAMPLE_TYPE == "low_variance":
        pf.resampler.resample_low_variance()
      else:
        print "Unrecognized resampling method: "+ pf.RESAMPLE_TYPE      
 
      pf.visualize() # Perform visualization

      end = time.time()
      i += 1
      times.append((i, (end - start)))
