#!/usr/bin/env python

import MotionModel
import numpy as np
import matplotlib.pyplot as plt
import itertools
import rospy
import rosbag
import utils

if __name__ == '__main__':
  model = 'odom'
  bag_path = '/home/nvidia/catkin_ws/src/lab1/bags/circle.bag'
  positions = []
  actual_pos = []
  num_iters = 1000
  if model == 'odom':
    init_top = '/initialpose' 
    odom_top = '/vesc/odom'
    infr_top = '/pf/ta/viz/inferred_pose'
    bag = rosbag.Bag(bag_path)
    # get initial pose message from bag
    init_msg = list(bag.read_messages(topics=init_top))[0][1]
    x,y = init_msg.pose.pose.position.x, init_msg.pose.pose.position.y
    yaw = utils.quaternion_to_angle(init_msg.pose.pose.orientation)
    particles = np.array((x,y,yaw)).reshape((1,3))
# TODO: remove noise
    odom_mm = MotionModel.OdometryMotionModel(particles, noise=False)   
    positions.append(tuple(odom_mm.particles[0][:2]))
    for i, tup in enumerate(bag.read_messages(topics=odom_top)):
      print odom_mm.particles
      topic, msg, t = tup 
      # apply movement msg
      odom_mm.motion_cb(msg)
      positions.append(tuple(odom_mm.particles[0][:2]))
      if i == num_iters:
        break
    for i, tup in enumerate(bag.read_messages(topics=infr_top)):
      topic, msg, t = tup 
      actual_pos.append((msg.pose.position.x, msg.pose.position.y)) 
      if i == num_iters:
        break
    xs, ys = zip(*positions)
    np.savetxt('/home/nvidia/olr_odom_positions_loop.txt', np.array(positions))
    np.savetxt('/home/nvidia/olr_odom_actual_loop.txt', np.array(actual_pos))
    plt.scatter(xs, ys, color='b', alpha=0.5, label='Odometry model')
    xs, ys = zip(*actual_pos)
    plt.scatter(xs, ys, color='r', alpha=0.5, label='Inferred pose')
    plt.title('Circle.bag odometry model vs. inferred pose')
    plt.axis('equal')
    plt.legend()
    plt.show()
  else: # kinematic model
    init_top = '/initialpose' 
    motion_top = "/vesc/sensors/core"
    servo_top = "/vesc/sensors/servo_position_command" 
    infr_top = '/pf/ta/viz/inferred_pose'
    bag = rosbag.Bag(bag_path)
    # set initial position
    init_msg = list(bag.read_messages(topics=init_top))[0][1]
    x,y = init_msg.pose.pose.position.x, init_msg.pose.pose.position.y
    yaw = utils.quaternion_to_angle(init_msg.pose.pose.orientation)
    particles = np.array((x,y,yaw)).reshape((1,3))
    kine_mm = MotionModel.KinematicMotionModel(particles, noise=False)   
    positions.append(tuple(kine_mm.particles[0][:2]))
    for i, tup in enumerate(bag.read_messages(topics=[motion_top, servo_top])):
      if tup.topic == motion_top:
        kine_mm.motion_cb(tup.message)
        positions.append(tuple(kine_mm.particles[0][:2]))
      elif tup.topic == servo_top:
        kine_mm.servo_cb(tup.message)
        if i == num_iters:
          break
    for i, tup in enumerate(bag.read_messages(topics=infr_top)):
      topic, msg, t = tup 
      actual_pos.append((msg.pose.position.x, msg.pose.position.y)) 
      if i == num_iters:
        break
    xs, ys = zip(*positions)
    np.savetxt('/home/nvidia/olr_kine_positions.txt', np.array(positions))
    np.savetxt('/home/nvidia/olr_kine_actual.txt', np.array(actual_pos))
    plt.scatter(xs, ys, color='b', alpha=0.5, label='Kinematic model')
    xs, ys = zip(*actual_pos)
    plt.scatter(xs, ys, color='r', alpha=0.5, label='Inferred pose')
    plt.title('Circle.bag odometry model vs. inferred pose')
    plt.axis('equal')
    plt.legend()
    plt.show()
