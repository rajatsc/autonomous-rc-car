#!/usr/bin/env python

import time
import sys
import rospy
import rosbag
import numpy as np
import utils as Utils

import torch
import torch.utils.data
from torch.autograd import Variable

from scipy import signal

from nav_msgs.srv import GetMap
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from KinematicMotionModel import KinematicMotionModel

import os.path

class MPPIController:

  def __init__(self, T, K, sigma=[0.5, 0.1], _lambda=0.5, testing=False):
    self.MIN_VEL = -0.75 #TODO make sure these are right
    self.MAX_VEL = 0.75
    self.MIN_DEL = -0.34
    self.MAX_DEL = 0.34
    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4614.0))
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5304))
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135))
    self.CAR_LENGTH = 0.33
    # config
    self.viz = False # visualize rollouts
    self.cont_ctrl = False # publish path continously

    self.testing = testing
    self.last_pose = None
    self.at_goal = True
  
    self.speed = 0
    self.steering_angle = 0
    self.prev_ctrl = None

    # MPPI params
    self.T = T # Length of rollout horizon
    self.K = K # Number of sample rollouts

    self._lambda = float(_lambda)

    self.goal_tensor = None

    # PyTorch / GPU data configuration
    # TODO
    # you should pre-allocate GPU memory when you can, and re-use it when
    # possible for arrays storing your controls or calculated MPPI costs, etc
    self.model = KinematicMotionModel()
    self.dtype = torch.cuda.FloatTensor
    print("Created Kinematic Motion Model:\n")
    print("Torch Datatype:", self.dtype)


    # initialize these once
    self.ctrl = torch.zeros((T,2)).cuda()

    self.sigma = torch.Tensor(sigma).type(self.dtype)
    self.inv_sigma = 1.0 / self.sigma #(2,)
    self.sigma = self.sigma.expand(self.T, self.K, 2) # (T,K,2)
    self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype) # (T,K,2)

    self.poses = torch.Tensor(self.K, self.T, 3).type(self.dtype) # (K,T,3)
    self.running_cost = torch.zeros(self.K).type(self.dtype) # (K,)
    self.bad_ks = torch.Tensor(self.K).type(self.dtype) #(K,)

    # control outputs
    self.msgid = 0

    # visualization paramters
    self.num_viz_paths = 40
    if self.K < self.num_viz_paths:
        self.num_viz_paths = self.K

    # We will publish control messages and a way to visualize a subset of our
    # rollouts, much like the particle filter
    self.ctrl_pub = rospy.Publisher(rospy.get_param("~ctrl_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_0"),
            AckermannDriveStamped, queue_size=2)
    self.path_pub = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)
    self.central_path_pub = rospy.Publisher("/mppi/path_center", Path, queue_size = 1)

    # Use the 'static_map' service (launched by MapServer.launch) to get the map
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
    self.map_info = map_msg.info # Save info about map for later use
    self.map_angle = -Utils.quaternion_to_angle(self.map_info.origin.orientation)
    self.map_c, self.map_s = np.cos(self.map_angle), np.sin(self.map_angle)
    print("Map Information:\n",self.map_info)

    # Create numpy array representing map for later use
    self.map_height = map_msg.info.height
    self.map_width = map_msg.info.width
    PERMISSIBLE_REGION_FILE = '/media/JetsonSSD/permissible_region'
    if os.path.isfile(PERMISSIBLE_REGION_FILE + '.npy'):
      self.permissible_region = np.load(PERMISSIBLE_REGION_FILE + '.npy')
    else:
      array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
      self.permissible_region = np.zeros_like(array_255, dtype=bool)
      self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                                # With values 0: not permissible, 1: permissible
      self.permissible_region = np.logical_not(self.permissible_region) # 0 is permissible, 1 is not

      KERNEL_SIZE = 31 # 15 cm = 7 pixels = kernel size 15x15
      kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))
      kernel /= kernel.sum()
      self.permissible_region = signal.convolve2d(self.permissible_region, kernel, mode='same') > 0 # boolean 2d array
      np.save(PERMISSIBLE_REGION_FILE, self.permissible_region)

    self.permissible_region = torch.from_numpy(self.permissible_region.astype(np.int)).type(torch.cuda.ByteTensor) # since we are using it as a tensor

    self.pose_pub = rospy.Publisher("/mppi/pose", PoseStamped, queue_size=10)

    print("Making callbacks")
    self.goal_sub = rospy.Subscriber("/move_base_simple/goal",
            PoseStamped, self.clicked_goal_cb, queue_size=1)
    self.pose_sub  = rospy.Subscriber("/pf/ta/viz/inferred_pose",
            PoseStamped, self.mppi_cb, queue_size=1)

  # TODO
  # You may want to debug your bounds checking code here, by clicking on a part
  # of the map and convincing yourself that you are correctly mapping the
  # click, and thus the goal pose, to accessible places in the map
  def clicked_goal_cb(self, msg):
    self.goal_tensor = torch.Tensor([msg.pose.position.x,
                          msg.pose.position.y,
                          Utils.quaternion_to_angle(msg.pose.orientation)]).type(self.dtype)
    self.at_goal = False
    print("SETTING Goal: ", self.goal_tensor.cpu().numpy())

  def cost(self, pose, goal, ctrl, noise, t):
    # input shapes: (K,3), (3,), (2,), (K,2)
    # This cost function drives the behavior of the car. You want to specify a
    # cost function that penalizes behavior that is bad with high cost, and
    # encourages good behavior with low cost.
    # We have split up the cost function for you to a) get the car to the goal
    # b) avoid driving into walls and c) the MPPI control penalty to stay
    # smooth
    # You should feel free to explore other terms to get better or unique
    # behavior

    cur_pose = pose.clone()

    # TODO: threshold the cost to 0 we are close enough
    ## FIRST - get the pose cost
    dist_to_goal = cur_pose - goal # (K,3)
    euclidian_distance = dist_to_goal[:,:2].norm(p=2, dim=1).pow(2)  # (K,)

    theta_distance = dist_to_goal[:,2]
    Utils.clamp_angle_tensor_(theta_distance)
    theta_distance.abs_()
    #self.pose_cost = euclidian_distance * 1.7 + theta_distance * 1.1 * (1.2 ** (t * 0.5))
    self.pose_cost = euclidian_distance * 1.7 + theta_distance * 1.05 * (1.2 ** (t * 0.5))

    ## SECOND - get the control cost, taking the abs
    ctrl_cost = self._lambda * ctrl * self.inv_sigma * noise # (K,2)
    ctrl_cost = ctrl_cost.abs().sum(dim=1) # (K,)

    ## THIRD - do a bounds check
    Utils.world_to_map_torch(cur_pose, self.map_info, self.map_angle, self.map_c, self.map_s) # cur pose is now in pixels

    xs = cur_pose[:,0]
    ys = cur_pose[:,1]
    self.bad_ks[:] = self.permissible_region[ys.long(), xs.long()] # (K,) with map value 0 or 1

    BOUNDS_COST = 10000
    bounds_check = self.bad_ks * BOUNDS_COST * t

    return self.pose_cost * 2.5 + ctrl_cost * 1.9 + bounds_check

  def mppi(self, init_pose):
    # init_pose (3,) [x, y, theta]
    # init_input (8,):
    #   0    1       2          3           4        5      6   7
    # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt
    t0 = time.time()

    self.running_cost.zero_()

    # convert pose into torch and one for each trajectory
    pose = init_pose.repeat(self.K, 1) # pose (K, 3)

    # MPPI should generate noise according to sigma
    torch.normal(0, self.sigma, out=self.noise)

    # Perform rollouts with those controls from your current pose
    for t in xrange(self.T):

      # Update nn_inputs with new pose information
      # combine noise with central control sequence
      noisy_ctrl = self.ctrl[t] + self.noise[t] # (2,) + (K,2)
      noisy_ctrl[:,0].clamp_(self.MIN_VEL, self.MAX_VEL)
      noisy_ctrl[:,1].clamp_(self.MIN_DEL, self.MAX_DEL)

      # Call model to learn new pose_dot
      self.model.particles = pose
      self.model.controls = noisy_ctrl
      self.model.ackerman_equations()

      # add new pose into poses
      self.poses[:,t,:] = pose.clone() # poses[:,t,:] (K,3) = pose (K,3)

      cur_cost = self.cost(pose, self.goal_tensor, self.ctrl[t], self.noise[t], t) # (K,3), (3,), (2,), (K,2) => (K,)
      self.running_cost += cur_cost

    # Perform the MPPI weighting on your calculatd costs
    # Scale the added noise by the weighting and add to your control sequence
    beta = torch.min(self.running_cost)
    self.running_cost -= beta
    self.running_cost /= -self._lambda
    torch.exp(self.running_cost, out=self.running_cost)
    weights = self.running_cost
    weights /= torch.sum(weights) # weights (K,)

    # apply weights to noise through time for each trajectory
    weights = weights.expand(self.T,self.K) # weights (T,K)
    weighted_vel_noise = weights * self.noise[:,:,0]  # (T,K) * (T,K) = (T,K)
    weighted_delta_noise = weights * self.noise[:,:,1]  # (T,K) * (T,K) = (T,K)

    # sum the weighted noise over all trajectories for each time step
    vel_noise_sum = torch.sum(weighted_vel_noise, dim=1)  # (T,)
    delta_noise_sum = torch.sum(weighted_delta_noise, dim=1)  # (T,)

    # update central control through time for vel and delta separately
    # self.ctrl # (T,2)
    self.ctrl[:,0] += vel_noise_sum  # (T,) += (T,)
    self.ctrl[:,1] += delta_noise_sum  # (T,) += (T,)
    self.ctrl[:,0].clamp_(self.MIN_VEL, self.MAX_VEL)
    self.ctrl[:,1].clamp_(self.MIN_DEL, self.MAX_DEL)


    # Notes: TODO
    # MPPI can be assisted by carefully choosing lambda, and sigma
    # It is advisable to clamp the control values to be within the feasible range
    # of controls sent to the Vesc
    # Your code should account for theta being between -pi and pi. This is
    # important.
    # The more code that uses pytorch's cuda abilities, the better; every line in
    # python will slow down the control calculations. You should be able to keep a
    # reasonable amount of calculations done (T = 40, K = 2000) within the 100ms
    # between inferred-poses from the particle filter.


    print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))
    return self.poses

  def get_control(self):
    # Apply the first control values, and shift your control trajectory
    run_ctrl = self.ctrl[0].clone()

    # shift all controls forward by 1, with last control replicated
    self.ctrl[:-1] = self.ctrl[1:]
    return run_ctrl

  def check_at_goal(self):
    if self.at_goal or self.last_pose is None:
      print 'Already at goal'
      return

    # TODO: if close to the goal, there's nothing to do
    XY_THRESHOLD = 0.45
    THETA_THRESHOLD = 0.3 # about 10 degrees
    difference_from_goal = (self.last_pose - self.goal_tensor).abs()
    xy_distance_to_goal = difference_from_goal[:2].norm()
    theta_distance_to_goal = difference_from_goal[2] % (2 * np.pi)
    if xy_distance_to_goal < XY_THRESHOLD and theta_distance_to_goal < THETA_THRESHOLD:
      print 'Goal achieved'
      self.at_goal = True
      self.speed = 0
      self.steering_angle = 0
      return

  def mppi_cb(self, msg):
    #print("callback")
    if self.last_pose is None:
      self.last_pose = torch.Tensor([msg.pose.position.x,
                                     msg.pose.position.y,
                                     Utils.quaternion_to_angle(msg.pose.orientation)]).type(self.dtype)
    if self.goal_tensor is None:
      # Default: initial goal to be where the car is when MPPI node is initialized
      self.goal_tensor = torch.Tensor([msg.pose.position.x,
                                     msg.pose.position.y,
                                     Utils.quaternion_to_angle(msg.pose.orientation)]).type(self.dtype)
      return

    theta = Utils.quaternion_to_angle(msg.pose.orientation)
    curr_pose = torch.Tensor([msg.pose.position.x,
                              msg.pose.position.y,
                              theta]).type(self.dtype)


    difference_from_goal = np.sqrt((curr_pose[0] - self.goal_tensor[0])**2 + 
                                   (curr_pose[1] - self.goal_tensor[1])**2) 
    if difference_from_goal < 1:
      self.MIN_VEL = -0.45 #TODO make sure these are right
      self.MAX_VEL = 0.45
    else:
      self.MIN_VEL = -.75 #TODO make sure these are right
      self.MAX_VEL = .75

    # don't do any goal checking for testing purposes
    self.last_pose = curr_pose
    self.pose_pub.publish(Utils.particle_to_posestamped(curr_pose, 'map'))

    poses = self.mppi(curr_pose)

    run_ctrl = None
    if not self.cont_ctrl:
      run_ctrl = self.get_control() 
      self.speed = run_ctrl[0]
      self.steering_angle = run_ctrl[1]
    
    if self.viz:
      self.visualize(poses)

    # For testing: send control into model and pretend this is the real location
    if self.testing:
      # TODO kinematic model can't handle just (3,)?
      self.model.particles = curr_pose.view(1,3)
      self.model.controls = torch.Tensor([self.speed, self.steering_angle]).type(self.dtype).view(1,2)
      self.model.ackerman_equations()

      return self.model.particles.view(3)

  def send_controls(self):
    if not self.at_goal:
      if self.cont_ctrl:
        run_ctrl = self.get_control()
        if self.prev_ctrl is None:
          self.prev_ctrl = run_ctrl
        speed = self.prev_ctrl[0] * .2 + run_ctrl[0] * .8
        steer = self.prev_ctrl[1] * .2 + run_ctrl[1] * .8
        self.prev_ctrl = (speed, steer)
      else:
        speed = self.speed 
        steer = self.steering_angle
      print("Speed:", speed, "Steering:", steer)
      ctrlmsg = AckermannDriveStamped()
      ctrlmsg.header.seq = self.msgid
      ctrlmsg.drive.steering_angle = steer
      ctrlmsg.drive.speed = speed
      self.ctrl_pub.publish(ctrlmsg)
      self.msgid += 1

  # Publish some paths to RVIZ to visualize rollouts
  def visualize(self, poses):
    # poses must be shape (self.num_viz_paths,T,3)
    if self.path_pub.get_num_connections() > 0:
      frame_id = 'map'
      for i in range(0, self.num_viz_paths):
        pa = Path()
        pa.header = Utils.make_header(frame_id)
        # poses[i,:,:] has shape (T,3)
        pa.poses = map(Utils.particle_to_posestamped, poses[i,:,:], [frame_id]*self.T)
        self.path_pub.publish(pa)

def test_MPPI(mp, N, goal=torch.Tensor([0.,0.,0.])):
  pose = torch.Tensor([0.,0.,0.]).type(mp.dtype)

  # Calculate costs for each of K trajectories
  mp.goal_tensor = goal.type(mp.dtype) # (3,)
  print("Start:", pose)
  mp.ctrl.zero_()

  # Initialize the mppi first pose
  mp.mppi_cb(Utils.particle_to_posestamped(pose, 'map'))

  for i in range(0,N):
    # ROLLOUT your MPPI function to go from a known location to a specified
    # goal pose. Convince yourself that it works.
    input_msg = Utils.particle_to_posestamped(pose, 'map')
    pose = mp.mppi_cb(input_msg)
    print("Now:", pose.cpu().numpy())
  print("End:", pose.cpu().numpy())

if __name__ == '__main__':

  T = 25
  K = 2000
  sigma = [0.4, 0.35] # TODO: These values will need to be tuned
  _lambda = .01

  rospy.init_node("mppi_control", anonymous=True) # Initialize the node

  # run with ROS
  mp = MPPIController(T, K, sigma, _lambda)
  rate = rospy.Rate(15)
  while not rospy.is_shutdown():
    mp.check_at_goal()
    mp.send_controls()
    rate.sleep()

  # test & DEBUG
 # mp = MPPIController(T, K, sigma, _lambda, testing=True)
 # test_MPPI(mp, 80, torch.Tensor([4.,3.,0.0]))
