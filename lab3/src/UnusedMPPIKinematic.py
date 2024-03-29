#!/usr/bin/env python

import time
import sys
import rospy
import rosbag
import numpy as np
import utils as Utils
from KinematicMotionModel import KinematicMotionModel

import torch
import torch.utils.data
from torch.autograd import Variable

from nav_msgs.srv import GetMap
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

class MPPIController:

  def __init__(self, T, K, sigma=[0.5, 0.1], _lambda=0.5, testing=False):
    self.MIN_VEL = -0.8 #TODO make sure these are right
    self.MAX_VEL = 0.8
    self.MIN_DEL = -0.34
    self.MAX_DEL = 0.34
    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4614.0))
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5304))
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135))
    self.CAR_LENGTH = 0.33

    self.testing = testing
    self.last_pose = None

    # MPPI params
    self.T = T # Length of rollout horizon
    self.K = K # Number of sample rollouts

    self._lambda = _lambda

    self.goal_tensor = None
    self.lasttime = None

    # PyTorch / GPU data configuration
    # TODO
    # you should pre-allocate GPU memory when you can, and re-use it when
    # possible for arrays storing your controls or calculated MPPI costs, etc
    self.dtype = torch.cuda.FloatTensor
    self.model = KinematicMotionModel()
    print("Created Kinematic Motion Model")
    print("Torch Datatype:", self.dtype)


    # initialize these once
    self.ctrl = torch.zeros((T,2)).cuda()

    self.sigma = torch.Tensor(sigma).type(self.dtype)
    self.inv_sigma = 1.0 / self.sigma #(2,)
    self.sigma = self.sigma.repeat(self.T, self.K, 1) # (T,K,2)
    self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype) # (T,K,2)

    self.poses = torch.Tensor(self.K, self.T, 3).type(self.dtype) # (K,T,3)
    self.running_cost = torch.zeros(self.K).type(self.dtype) # (K,)

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
    # rotation
    self.map_angle = -Utils.quaternion_to_angle(self.map_info.origin.orientation)
    self.map_c, self.map_s = np.cos(self.map_angle), np.sin(self.map_angle)
    print("Map Information:\n",self.map_info)

    # Create numpy array representing map for later use
    self.map_height = map_msg.info.height
    self.map_width = map_msg.info.width
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                              # With values 0: not permissible, 1: permissible
    self.permissible_region = np.logical_not(self.permissible_region) # 0 is permissible, 1 is not
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
    print("Current Pose: ", self.last_pose)
    print("SETTING Goal: ", self.goal_tensor)

  def cost(self, pose, goal, ctrl, noise, t):
    # input shapes: (K,3), (3,), (2,), (K,2), (1)
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
    pose_cost = euclidian_distance * 4.5 + theta_distance * 0.8 # (K,)

    ## SECOND - get the control cost, taking the abs
    ctrl_cost = self._lambda * ctrl * self.inv_sigma * noise # (K,2)
    ctrl_cost = ctrl_cost.abs().sum(dim=1) # (K,)

    ## THIRD - do a bounds check
    Utils.world_to_map_torch(cur_pose, self.map_info, self.map_angle, self.map_c, self.map_s) # cur pose is now in pixels

    xs = cur_pose[:,0]
    ys = cur_pose[:,1]
    bad_ks  = (xs < 0)
    bad_ks |= (xs >= self.map_width)  # byte tensors, NOT indices
    bad_ks |= (ys < 0)
    bad_ks |= (ys >= self.map_height) # byte tensors, NOT indices
    xs *= (~bad_ks).float() # (K,) modified bad coords to be in range, so we can index the map
    ys *= (~bad_ks).float() # (K,) modified bad coords to be in range, so we can index the map

    map_values = self.permissible_region[ys.long(), xs.long()] # (K,) with map value 0 or 1
    bad_ks |= map_values

    BOUNDS_COST = 10000000
    bounds_check = bad_ks.type(self.dtype) * BOUNDS_COST


    return pose_cost * (0.9 ** t) + bounds_check + ctrl_cost

  # Process input the same way we did before training the model
  # Will need to: (x_dot, y_dot, theta_dot) / dt * 0.1
  def process_nn_input(self, init_input):
    # init_input (8,)
    init_input[2] = Utils.clamp_angle(init_input[2])
    return init_input

  def mppi(self, init_pose):
    # init_pose (3,) [x, y, theta]
    dt = 0.1
    t0 = time.time()
    init_pose = torch.from_numpy(init_pose.astype(np.float32)).type(self.dtype)
    self.running_cost.zero_()
    # convert pose into torch and one for each trajectory
    pose = init_pose.repeat(self.K, 1) # pose (K, 3)

    # MPPI should generate noise according to sigma
    torch.normal(0, self.sigma, out=self.noise)

    # Perform rollouts with those controls from your current pose
    for t in xrange(self.T):
      # combine noise with central control sequence
      noisy_ctrl = self.ctrl[t] + self.noise[t] # (2,) + (K,2)
      noisy_ctrl[:,0].clamp_(self.MIN_VEL, self.MAX_VEL)
      noisy_ctrl[:,1].clamp_(self.MIN_DEL, self.MAX_DEL)


      # Call model to learn new pose
      # pose is modified in place
      self.model.particles = pose.cpu().numpy()
      self.model.controls = noisy_ctrl.cpu().numpy()
      self.model.ackerman_equations()
      pose = torch.from_numpy(self.model.particles.astype(np.float32)).type(self.dtype)

      # add new pose into poses
      self.poses[:,t,:] = pose.clone() # poses[:,t,:] (K,3) = pose (K,3)

      # Calculate costs for each of K trajectories
      cur_cost = self.cost(pose, self.goal_tensor, self.ctrl[t], self.noise[t], t) # (K,3), (3,), (2,), (K,2) => (K,)
      self.running_cost += cur_cost

    # Perform the MPPI weighting on your calculatd costs
    # Scale the added noise by the weighting and add to your control sequence

    beta = torch.min(self.running_cost)
    weights = torch.exp((-1.0 / self._lambda) * (self.running_cost - beta))
    eta = torch.sum(weights)
    weights /= eta # weights (K,)

    # apply weights to noise through time for each trajectory
    weights = weights.repeat(self.T,1) # weights (T,K)
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
    # print "self.ctrl end of mppi",self.ctrl

    # Apply the first control values, and shift your control trajectory
    run_ctrl = self.ctrl[0].clone()

    # shift all controls forward by 1, with last control replicated
    self.ctrl[:-1] = self.ctrl[1:]

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
    return run_ctrl, self.poses

  def mppi_cb(self, msg):
    #print("callback")
    if self.last_pose is None:
      self.last_pose = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 Utils.quaternion_to_angle(msg.pose.orientation)])
      # Default: initial goal to be where the car is when MPPI node is
      # initialized
      # if we are testing, keep the goal that we passed in!!
      if not self.testing:
        self.goal_tensor = torch.from_numpy(self.last_pose.astype(np.float32)).type(self.dtype)
      self.lasttime = msg.header.stamp.to_sec()
      return

    theta = Utils.quaternion_to_angle(msg.pose.orientation)
    curr_pose = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          theta])

    # TODO: if close to the goal there's nothing to do
    # XY_THRESHOLD = 0.05
    # THETA_THRESHOLD = 0.17 # about 10 degrees
    # difference_from_goal = curr_pose - self.goal_tensor
    # xy_distance_to_goal = np.linalg.norm(difference_from_goal[:2])
    # theta_distance_to_goal = np.abs(difference_from_goal[2])
    # if xy_distance_to_goal < XY_THRESHOLD and theta_distance_to_goal < THETA_THRESHOLD:
    #   print 'Close to goal'
    #   return

    self.last_pose = curr_pose

    self.pose_pub.publish(Utils.particle_to_posestamped(curr_pose, 'map'))

    run_ctrl, poses = self.mppi(curr_pose)

    self.send_controls( run_ctrl[0], run_ctrl[1] )

    self.visualize(poses)

    ## For testing: send control into model and pretend this is the real location
    if self.testing:
      self.model.particles = curr_pose.reshape(1,3)
      self.model.controls = run_ctrl.cpu().numpy().reshape(1,2)
      self.model.ackerman_equations()

      return self.model.particles.reshape(3)

  def send_controls(self, speed, steer):
    # print("Speed:", speed, "Steering:", steer)
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

def test_MPPI(mp, N, goal=np.array([0.,0.,0.])):
  init_input = np.array([0.,0.,0.,0.,1.,0.,0.,0.])
  pose = np.array([3.2,3.5,0.6])
  mp.goal_tensor = torch.from_numpy(goal.astype(np.float32)).type(mp.dtype)
  # print("Start:", pose)
  mp.ctrl.zero_()

  # Initialize the mppi first pose
  mp.mppi_cb(Utils.particle_to_posestamped(pose, 'map'))

  for i in range(0,N):
    # ROLLOUT your MPPI function to go from a known location to a specified
    # goal pose. Convince yourself that it works.
    input_msg = Utils.particle_to_posestamped(pose, 'map')
    pose = mp.mppi_cb(input_msg)
    # print("Now:", pose)
  # print("End:", pose)

if __name__ == '__main__':

  T = 20
  K = 1000
  sigma = [1, 0.5] # TODO: These values will need to be tuned
  _lambda = 2.0

  # run with ROS
  rospy.init_node("mppi_control", anonymous=True) # Initialize the node
  #mp = MPPIController(T, K, sigma, _lambda)
  #rospy.spin()

  # test & DEBUG
  mp = MPPIController(T, K, sigma, _lambda, testing=True)
  # test_MPPI(mp, 10, np.array([0.,0.,0.]))
  test_MPPI(mp, 20, np.array([3.8,6.46,2.232]))

  # 2. improve performance
  #    - replace all the np. to torch when possible
  # 3. improve cost function (threshold if close enough?)


# put model on origin, generate rollouts, do any go backwards?
# try MPPI with kinematic model, rather than NN model


# TODO
# LOGIC in MPPI is good
# SOMETHING WRONG WITH BOUND CHECKING (somethings flipped??)
# THings we need to fix: cost function, model
