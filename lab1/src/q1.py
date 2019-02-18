#!/usr/bin/env python

import MotionModel
import numpy as np
import matplotlib.pyplot as plt
import itertools


def Plots_Odometry():
  particle = np.zeros((1000,3))
  odom_mm = MotionModel.OdometryMotionModel(particle)   
  num_iters = 2
  colors = itertools.cycle(["r", "r", "b", "b", "g", "g"])
  noises = [0.1, 0.5]
  lables = ["low variance (0.1)", "high variance (0.5)"]
  plt.scatter(0,0, color=next(colors))
  plt.scatter(particle[:,0], particle[:,1], color=next(colors))
  for j, noise in enumerate(noises):
    particle[:] = np.zeros((1000,3))
    for i in range(num_iters):
      odom_mm.SD_ODOM_THETA = noise
      control = np.array((2,0,0))
      odom_mm.apply_motion_model(particle, control)
      if i == 0:
        plt.scatter(particle[:,0], particle[:,1], color=next(colors), alpha=0.1,  label=lables[j])
      else:
        plt.scatter(particle[:,0], particle[:,1], color=next(colors), alpha=0.1)
  plt.legend()
  plt.show()


def Plots_Kinematic():
  particle=np.zeros(1000,3)
  num_iters=2
  control=np.array((1,1))

  km_mm=MotionModel.KinematicMotionModel(particle)
  km_mm.final_control=control

  vel_noises=[0.1, 0.2]
  steer_angle_noises=[0.1]

  ##Colors and Labels for plotting
  colors_arr= np.chararray((len(vel_noises),len(steer_angle_noises)))
  labels_arr= np.chararray((len(vel_noises),len(steer_angle_noises)))

  colors_arr[0,0]='r'
  colors_arr[1,0]='y'

  for j in range(len(vel_noises)):
    for k in range(len(steer_angle_noises)):

      labels_arr[j,k]='speed noise ='+str(vel_noises[i])+', steering angle noise ='+str(steer_angle_noises[j])



  

  for j, v_noise in enumerate(vel_noises):
    for k, s_noise in enumerate(steer_angle_noises):
      
      particle[:] = np.zeros((1000,3))
      km_mm.particles=particle
      km_mm.SD_KM_VEL=vel_noise
      km_mm.SD_KM_STEER_ANGLE=steer_angle_noise

      

      for i in num_iters:
        km_mm.ackerman_equations()
        plt.scatter( particle[:,0], particle[:,1], color=colors_arr[j,k], label=labels_arr[j,k] if i == 0 else "")



  plt.legend()
  plt.show()



      





if __name__ == '__main__':
  
  Plots_Kinematic()
