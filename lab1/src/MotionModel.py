#!/usr/bin/env python

from __future__ import division
import rospy
import utils
import math
import numpy as np
import utils as Utils
from std_msgs.msg import Float64
from threading import Lock
import message_filters
from vesc_msgs.msg import VescStateStamped


class OdometryMotionModel:

  def __init__(self, particles, state_lock=None, noise=True):
    self.last_pose = None # The last pose that was received
    self.particles = particles
    self.noise = noise
    # TODO: tune me!!
    self.SD_ODOM_POS = 0.035    # Standard deviation for Gaussian noise
    self.SD_ODOM_THETA = 0.06
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
    
  def motion_cb(self, msg):
    new_pose = np.zeros((3,1)) 
    new_pose[0] = msg.pose.pose.position.x
    new_pose[1] = msg.pose.pose.position.y
    new_pose[2] = utils.quaternion_to_angle(msg.pose.pose.orientation)
    
    self.state_lock.acquire()
    if isinstance(self.last_pose, np.ndarray):
      dx = new_pose[0,0] - self.last_pose[0,0]
      dy = new_pose[1,0] - self.last_pose[1,0]
      d_theta = new_pose[2,0] - self.last_pose[2,0]
      # Compute the control from the msg and last_pose
      # Apply noise. 
      # TODO: May want to scale e_theta
      # Rotation matrix
      rotation = utils.rotation(self.last_pose[2,0])
      dX = np.array((dx, dy)).reshape((2,1))
      dX_prime = np.matmul(rotation.transpose(), dX)
      control = np.array((dX_prime[0,0], dX_prime[1,0], d_theta))

      self.apply_motion_model(self.particles, control)

    self.last_pose = new_pose
    self.state_lock.release()
    
  def apply_motion_model(self, proposal_dist, control):
    # Update the proposal distribution by applying the control to each particle
    # YOUR CODE HERE
    if self.noise:
      noise = np.random.normal([0,0,0], [self.SD_ODOM_POS, self.SD_ODOM_POS, self.SD_ODOM_THETA], proposal_dist.shape)
    else:
      noise = np.zeros(proposal_dist.shape)
    noisy_control = noise + control

    c, s = np.cos(proposal_dist[:,2]), np.sin(proposal_dist[:,2])
    self.particles[:,0] += c * noisy_control[:,0] - s * noisy_control[:,1]
    self.particles[:,1] += s * noisy_control[:,0] + c * noisy_control[:,1]
    self.particles[:,2] = proposal_dist[:,2] + noisy_control[:,2]

class KinematicMotionModel:

  def __init__(self, particles, state_lock=None, noise=True, fixed_dt=0):  ##Add particles and state lock here
    #self.last_servo_cmd = None # The most recent servo command
    #self.last_vesc_stamp = None # The time stamp from the previous vesc state msg
    """
    self.old_position_x=0
    self.old_position_y=0
    self.old_heading_angle_theta=0
    """

    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock

    self.particles=particles
    self.old_vesc_time_stamp=None  ##Take None
    self.new_vesc_time_stamp=None
    self.noise=noise
    self.fixed_dt = fixed_dt

    self.SD_km_speed=1.0
    self.SD_km_steer_angle=0.5

    self.final_control=np.zeros(2)
    self.L=.33 # TODO: set actual car length
    self.servo_pos_topic = "/vesc/sensors/servo_position_command" 
    #self.particles = particles
    
    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset")) # Offset conversion param from rpm to speed
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain"))   # Gain conversion param from rpm to speed
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset")) # Offset conversion param from servo position to steering angle
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain")) # Gain conversion param from servo position to steering angle
    """
    self.SPEED_TO_ERPM_OFFSET=0
    self.SPEED_TO_ERPM_GAIN=4614
    self.STEERING_TO_SERVO_OFFSET=0.5304
    self.STEERING_TO_SERVO_GAIN=-1.2135
    """


    #Crude Implementation

    self.servo_pos_sub = rospy.Subscriber(self.servo_pos_topic, Float64, self.servo_cb, queue_size=1)

  def servo_cb(self, raw_msg_val):
      self.state_lock.acquire()
      control_delta=(raw_msg_val.data-self.STEERING_TO_SERVO_OFFSET)/self.STEERING_TO_SERVO_GAIN
      self.final_control[1]=control_delta
      self.state_lock.release()
  
  def motion_cb(self, raw_msg_val):
      control_vel=(raw_msg_val.state.speed-self.SPEED_TO_ERPM_OFFSET)/self.SPEED_TO_ERPM_GAIN
      self.final_control[0]=control_vel
      self.new_vesc_time_stamp=raw_msg_val.header.stamp
      self.ackerman_equations()

  def ackerman_equations(self):

    speed=self.final_control[0]
    omega=self.final_control[1]

    # add noise
    if self.noise:
      speed=speed+np.random.normal(loc=0, scale=self.SD_km_speed, size=self.particles.shape[0])
      omega=omega+np.random.normal(loc=0, scale=self.SD_km_steer_angle, size=self.particles.shape[0])

    if self.old_vesc_time_stamp is None:
      self.old_vesc_time_stamp = self.new_vesc_time_stamp
      return

    self.state_lock.acquire()
    if self.fixed_dt > 0:
      detal_t = self.fixed_dt
    else:
      delta_t = self.new_vesc_time_stamp.to_sec() - self.old_vesc_time_stamp.to_sec()

    
    # update theta
    if self.final_control[1] == 0: # if steering angle is zero
      self.particles[:,0] += speed * np.cos(self.particles[:,2]) * delta_t
      self.particles[:,1] += speed * np.sin(self.particles[:,2]) * delta_t
    else: # non-zero steering angle, integrate forward
      # get sin and cos of theta
      s_theta = np.sin(self.particles[:,2])
      c_theta = np.cos(self.particles[:,2])
      
      # update theta
      #sin_2_beta = 4*np.tan(omega) / (4+np.square(np.tan(omega)))
      sin_2_beta = np.sin(2.0*np.arctan(np.tan(omega)/2.0))
      delta_theta = (speed/self.L) * sin_2_beta * delta_t    
      self.particles[:,2] += delta_theta 
      #self.particles[:,2] %= 2 * np.pi
    
      # get sin and cos of new theta
      s_theta_one = np.sin(self.particles[:,2])
      c_theta_one = np.cos(self.particles[:,2])

      # update particle positions
      self.particles[:,0] += (self.L / sin_2_beta) * (s_theta_one - s_theta)
      self.particles[:,1] += (self.L / sin_2_beta) * (-c_theta_one + c_theta)
   # else:
   #   print "straight"

    self.state_lock.release()

    """
    self.old_position_x=self.new_position_x
    self.old_position_y=self.new_position_y
    self.old_heading_angle_theta=self.new_heading_angle_theta
    """
    self.old_vesc_time_stamp=self.new_vesc_time_stamp


  """
  def Synced_kinematic():

  rospy.init_node('Synced_kinematic')
  servo_sub=message_filters.Subscriber('', )
  speed_sub=message_filters.Subscriber('', )

  ts=message_filters.ApproximateTimeSynchronizer([servo_sub, depth_sub], 10, 1)
  ts=registerCallback(synched_callback)
  try:
    rospy.spin()
  except KeyboardInterrupt
    print "Shutting down"

  def Synced_Callback(servo,speed):
    rospy.loginfo("got synced messages")
  """

  
if __name__ == '__main__':
  #servo_pos_topic=rospy.get_param("servo_pos_topic")
  #speed_pos_topic=rospy.get_param("speed_pos_topic")
  servo_pos_topic="/vesc/sensors/servo_position_command" 
  speed_topic="/vesc/sensors/core"
  add_noise=False
  len_car=1

  motion_model = my_KinematicMotionModel=KinematicMotionModel(np.zeros((100,3)))
  rospy.init_node('my_KinematicMotionModel', anonymous=True)
  motion_sub = rospy.Subscriber("/vesc/sensors/core", VescStateStamped, motion_model.motion_cb, queue_size=1)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
