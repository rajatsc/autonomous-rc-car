#!/usr/bin/env python

import MotionModel
import numpy as np
import matplotlib.pyplot as plt
import itertools
import rospy
import rosbag
import utils
import sys

# usage: python q3.py odom circle [print]

if __name__ == '__main__':
  # get args
  MODEL = sys.argv[1]
  bag = sys.argv[2]
  BAG_PATH = '/home/nvidia/catkin_ws/src/lab1/bags/{}.bag'.format(bag)
  PRINT = False
  PLOT_EXPECTED = False
  if len(sys.argv) > 3:
    if 'print' in sys.argv:
      PRINT = True
    if 'plot_expected' in sys.argv:
      PLOT_EXPECTED = True

  # start from somewhere that's actually in the circle
  START_I = 75
  # q3 plot noise TODO
  #NUM_ITERS = 20
  # q2 open loop
  NUM_ITERS = -1

  # q3 TODO
  #NUM_PARTICLES = 500
  #NOISE = True
  # q2 open loop
  NUM_PARTICLES = 1
  NOISE = False

  colors = itertools.cycle(["r", "g", "b"])

  # topics
  INIT_TOP = '/initialpose'
  ODOM_TOP = '/vesc/odom'
  MOTION_TOP = '/vesc/sensors/core'
  SERVO_TOP = '/vesc/sensors/servo_position_command'
  INFR_TOP = '/pf/ta/viz/inferred_pose'

  bag = rosbag.Bag(BAG_PATH)
  
  # expected route
  expected_msgs = list(bag.read_messages(topics=INFR_TOP))
  expected_msgs = expected_msgs[START_I:]

  # plot expected route
  if PLOT_EXPECTED:
    if PRINT:
      print 'EXPECTED'
    for i, tup in enumerate(expected_msgs):
      topic, msg, t = tup
      ex = msg.pose.position.x
      ey = msg.pose.position.y
      # q3 plot noise TODO
      #plt.scatter([ex], [ey], color=next(colors))
      # q2 open loop
      plt.scatter([ex], [ey], color='r')
      #plt.annotate(str(i), (ex, ey))

      if PRINT:
        print (ex, ey)

      if i == NUM_ITERS:
        break

  # get init pose
  init_msg = expected_msgs[0]
  init_x, init_y = (init_msg[1].pose.position.x, init_msg[1].pose.position.y)
  init_theta = utils.quaternion_to_angle(init_msg[1].pose.orientation)
  init_time = init_msg[2]
  # plot init pose
  xs = [init_x] * NUM_PARTICLES
  ys = [init_y] * NUM_PARTICLES
  plt.scatter(xs, ys, color='k', alpha=0.5)
  if PRINT:
    print 'OURS ' + MODEL
    print 't0'
    print xs
    print ys

  # populate all particles to have the initial pose
  particles = np.zeros((NUM_PARTICLES, 3))
  particles[:,0] = init_x
  particles[:,1] = init_y
  particles[:,2] = init_theta

  if MODEL == 'odom':
    mm = MotionModel.OdometryMotionModel(particles, noise=NOISE)
    mm.SD_ODOM_POS = 0.0005
    mm.SD_ODOM_THETA = 0.05
    bag_msgs = list(bag.read_messages(topics=ODOM_TOP))
    # find the first msg that starts after init_time
    start_i = (i for i, tup in enumerate(bag_msgs) if tup[2] >= init_time).next()
    bag_msgs = bag_msgs[start_i:]

  else: # kinematic TODO
    mm = MotionModel.KinematicMotionModel(particles)
    #mm.SD_km_speed = 0.03
    #mm.SD_km_steer_angle = 0.2
    #bag_msgs = list(bag.read_messages(topics=[MOTION_TOP, SERVO_TOP]))
    #start_i = (i for i, tup in enumerate(bag_msgs) if tup[0] == MOTION_TOP and tup[1].state.speed != 0).next()
  
  for i, tup in enumerate(bag_msgs):
    top = bag_msgs[i].topic
    msg = bag_msgs[i].message

    if MODEL == 'odom' or top == MOTION_TOP:
      mm.motion_cb(msg)
    else:
      mm.servo_cb(msg)

    new_pos = [tuple(mm.particles[p,:2]) for p in range(NUM_PARTICLES)]
      
    xs, ys = zip(*new_pos)
    # q3 noise TODO
    #plt.scatter(xs, ys, color=next(colors), alpha=0.5, marker='.')
    # q2 open loop
    plt.scatter(xs, ys, color='b', alpha=0.5, marker='.')
    #plt.annotate("D"+str(i), (xs[0], ys[0]))

    if i == NUM_ITERS:
      break
    
    if PRINT:  
      print 't' + str(i)
      print xs
      print ys
  
  plt.show()


