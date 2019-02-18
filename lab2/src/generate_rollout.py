#!/usr/bin/env python

import sys
import cv2
import utils
import numpy as np
import tf
import time
import rospy
import matplotlib.pyplot as plt
from KinematicMotionModel import KinematicMotionModel

CONTROL_V = 0.5
DELTA_T = 0.01 
NUM_ITERS = 300
NUM_ANGLES = 50
MIN_ANGLE = -0.34
MAX_ANGLE = 0.34
CAMERA_K = np.matrix([617.384521484375, 0.0, 320.24609375, 
                      0.0, 615.8568115234375, 242.38568115234375, 
                      0.0, 0.0, 1.0]).reshape((3,3))
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
DOWNSAMPLE = 1.0
PW = 2
ROI_HEIGHT = 130
ROI_OFFSET = 50

if __name__ == '__main__':
  rospy.init_node('rollout')
  transformer = tf.TransformListener()
  print 'before sleep'
  time.sleep(5)
  rospy.init_node('rollout')
  transformer = tf.TransformListener()
  print 'done waiting'

  position, quaternion = transformer.lookupTransform('/camera_rgb_optical_frame', 
                                                     '/kinematic', 
                                                     transformer.getLatestCommonTime(
                                                        '/camera_rgb_optical_frame', '/kinematic'))
  rot = tf.transformations.quaternion_matrix(quaternion)
  rot[:-1,3] = position

  all_temp_img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
  angles_to_try = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_ANGLES)
  plot_num = 0
  for angle in angles_to_try:
    init_controls = np.array([CONTROL_V, angle])
    particles = np.zeros((1,3))
    km = KinematicMotionModel(particles, noise=False, fixed_dt=DELTA_T, init_controls=init_controls)
    
    # coordinates in camera image
    xs = []
    ys = []
    # coordinates in the 3D kinematic frame
    world_xs = []
    world_ys = []
    # template image
    template_img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    min_x = IMAGE_WIDTH
    max_x = 0
    
    # iterate over time
    for i in xrange(NUM_ITERS):
      # apply kinematic
      km.ackerman_equations()
      x, y, theta = km.particles[0]
      world_xs.append(x)
      world_ys.append(y)
      
      # get x,y in 3D camera frame
      camera_frame = np.matmul(rot, np.array([x,y,0,1]).transpose())
      # ignore if point is behind the camera
      if camera_frame[2] < 0:
        continue
      camera_frame = (camera_frame[:3]/camera_frame[2]).reshape(3,1)
      # get x,y in camera image 2D frame
      pixel_frame = np.matmul(CAMERA_K, camera_frame)
      # ignore if point is outside of image
      if pixel_frame[0] < 0 or pixel_frame[0] >= IMAGE_WIDTH or \
         pixel_frame[1] < 0 or pixel_frame[1] >= IMAGE_HEIGHT:
        continue
      
      xs.append(float(pixel_frame[0]))
      ys.append(float(pixel_frame[1]))
      
      if pixel_frame[0] > max_x:
        max_x = pixel_frame[0]

      if pixel_frame[0] < min_x:
        min_x = pixel_frame[0]

      # fill in template img at x, y TODO edge
      template_img[pixel_frame[1]-PW:pixel_frame[1]+PW,
                   pixel_frame[0]-PW:pixel_frame[0]+PW] = 1

    # dont bother plotting if no points made it in the image
    if len(xs) == 0 or len(ys) == 0:
      continue

    plot_num += 1
    max_x += 5
    min_x -= 5
    if max_x > IMAGE_WIDTH:
      max_x = IMAGE_WIDTH
    if min_x < 0:
      min_x = 0
    scaled_img = cv2.resize(template_img, (0,0), fx=DOWNSAMPLE, fy=DOWNSAMPLE)
    scaled_img = (scaled_img*255).repeat(3).reshape(scaled_img.shape[0], scaled_img.shape[1], 3)
    roi, x, y = utils.get_roi_center_naiive(scaled_img, ROI_HEIGHT, ROI_OFFSET)  
    cropped_img = scaled_img[-ROI_HEIGHT-ROI_OFFSET:-ROI_OFFSET,:]
    
    # add to all_temp_img
    all_temp_img = cv2.bitwise_or(all_temp_img, template_img)

    # save template images
    cv2.imwrite("temp{}_{}_{}.png".format(plot_num, CONTROL_V, angle), cropped_img)

    # save template center coordinates
    np.save("temp{}_{}_{}".format(plot_num, CONTROL_V, angle), (x,y))
    #np.save("{}_{}".format(CONTROL_V, angle), cropped_img)
    print "temp{} saved".format(plot_num)

    # plot for kinematic
    #print "({})world_xs={}".format(plot_num, world_xs)
    #print "({})world_ys={}".format(plot_num, world_ys)
    #plt.plot(world_xs, world_ys)


  # kinematic plot
  #plt.savefig('kinematic.png')
  #plt.show()

  # all template img
  cv2.imwrite('all_templates.png', all_temp_img*255)



