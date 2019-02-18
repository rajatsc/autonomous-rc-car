#!/usr/bin/env python

from __future__ import division
import rospy
import utils
import math
import numpy as np
import torch
import utils as Utils
from std_msgs.msg import Float64
from threading import Lock
import message_filters

class KinematicMotionModel:

  def __init__(self):  ##Add particles and state lock here
    self.particles= None # (k,3)
    self.fixed_dt = 0.1
    self.controls = None  # (K,2)

    self.L=.33 # TODO: set actual car length
    self.x = torch.Tensor([0]).type(torch.cuda.LongTensor)
    self.y = torch.Tensor([1]).type(torch.cuda.LongTensor)
    self.theta = torch.Tensor([2]).type(torch.cuda.LongTensor)
    

  def ackerman_equations(self):
    speed = self.controls[:,0]  # (K,)
    omega = self.controls[:,1]  # (K,)
    zero_del_indices = omega == 0 # (K,) byte tensor
    zero_del_index_values = zero_del_indices.nonzero()  # (K,) long tensor of index values
    zero_del_index_values = zero_del_index_values.view(zero_del_index_values.nelement())
    nonzero_indices = (~zero_del_indices).nonzero()
    nonzero_indices = nonzero_indices.view(nonzero_indices.nelement())

    # Steering angle is zero case
    if zero_del_index_values.nelement() > 0:
        self.particles[zero_del_index_values, self.x] += speed[zero_del_index_values] * torch.cos(self.particles[zero_del_index_values, theta]) * self.fixed_dt
        self.particles[zero_del_index_values, self.y] += speed[zero_del_index_values] * torch.sin(self.particles[zero_del_index_values, self.theta]) * self.fixed_dt

    # update theta
    # if self.final_control[1] == 0: # if steering angle is zero
    #   self.particles[:,0] += speed * np.cos(self.particles[:,2]) * self.fixed_dt
    #   self.particles[:,1] += speed * np.sin(self.particles[:,2]) * self.fixed_dt

    if nonzero_indices.nelement() > 0:
        # Steering angle is NOT zero
        # get sin and cos of theta
        s_theta = torch.sin(self.particles[nonzero_indices, self.theta])  # (K,)
        c_theta = torch.cos(self.particles[nonzero_indices, self.theta])  # (K,)
        
        sin_2_beta = torch.sin(2.0*torch.atan(torch.tan(omega[nonzero_indices])/2.0))  # (K,)
        delta_theta = (speed[nonzero_indices]/self.L) * sin_2_beta * self.fixed_dt  # (K,) 
        self.particles[nonzero_indices, self.theta] += delta_theta  # (K,)
        #self.particles[:,2] %= 2 * np.pi
      
        # get sin and cos of new theta
        s_theta_one = torch.sin(self.particles[nonzero_indices, self.theta])  # (K,)
        c_theta_one = torch.cos(self.particles[nonzero_indices, self.theta])  # (K,)

        # update particle positions
        delta_x = (self.L / sin_2_beta) * (s_theta_one - s_theta)  # (K,)
        delta_y = (self.L / sin_2_beta) * (-c_theta_one + c_theta)  # (K,)
        self.particles[nonzero_indices, x] += delta_x
        self.particles[nonzero_indices, y] += delta_y
        return (delta_x, delta_y, delta_theta)
