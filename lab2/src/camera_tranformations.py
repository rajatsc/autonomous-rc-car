from __future__ import division
import numpy as np


#world_points=(4Xn) where n is the number of points
#and the each column array is (U,V,W,1) where 
#U,V and W are the x, y and z coordinates in the 
#world coordinate frame respectively

#camera_points=(4Xn) Each column array is (X,Y,Z,1)
#where X,Y and Z are the x,y and z coordinates in
#the camera frame respectively.

#film_points=(3Xn) Each column array is (x', y', z')
#where x'/z' and y'/z' are the x and y coordinates
# in the film frame respectively.

#pixel_points=(3Xn) Each column is (u',v',w') where 
#u'/w' and v'/w' are the x and y coordinates in the
#pixel frame respectively

#tran is a 1-D array representing the cooridinates of 
#origin of Camera coordinate frame in World coordinate frame

#f is the depth of the film(i.e image frame) 
#in the camera coordinate frame

#s_x/s_y is the aspect ratio

#(o_x,o_y) is the image centre or principal point


def world2pixel(world_points):   #Forward projection
	camera_points=world2camera(world_points)
	film_points=camera2film(camera_points)
	pixel_points=film2pixel(film_points)
	return pixel_points

def pixel2world(pixel_points):   #Backward Projection
	#Some code 



def world2camera(alpha, beta, gamma, tran, world_points):
	final_rot_mat=create_rotational_matrix(rot)
	final_tran_mat=create_translation_matrix(tran)
	camera_points=final_rot_mat*final_tran_mat*np.mat(world_points)
	return camera_points

def camera2film(f, camera_points):

	temp_arr=np.diag(v=np.array([f,f,1,0]),k=0)
	perspective_mat=np.mat(temp_arr[0:3,:]
	film_points=perspective_mat*np.mat(camera_points)
	return film_points

def film2pixel(s_x,s_y,o_x,o_y, film_points):
	affine_mat=create_affine_matrix(s_x, s_y, o_x, o_y)
	pixel_points=affine_mat*np.mat(film_points)
	return pixel_points



###############################################
#Camera Extrinsics 

def create_rotational_matrix(alpha, beta, gamma):

	#alpha, beta and gamma are the improper Euler angles 
	#about axes x,y and z 
	rot_alpha_x=np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)],[0, np.sin(alpha), np.cos(alpha)]])
	rot_beta_y=np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0, np.cos(beta)]])
	rot_gamma_z=np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma), np.cos(gamma),0],[0,0,1]])
	rot=np.mat(rot_gamma_z)*np.mat(rot_beta_y)*np.mat(rot_alpha_x)

	temp_rot=np.zeros((4,4))
	temp_rot[0:3,0:3]=rot
	temp_rot[3,3]=1
	final_rot_mat=np.matrix(temp_rot)
	return final_rot_mat


def create_translation_matrix(tran):
	temp_tran=np.diag(v=np.ones(4),k=0)
	temp_tran[0:3,3]=tran
	final_tran_mat=np.matrix(temp_tran)
	return final_tran_matrix


##################################################
#Camera Intrinsics

def create_affine_matrix(s_x,s_y,o_x,o_y):

	affine=np.array([1/s_x,0,o_x],[0,1/s_y,o_y],[0,0,1])
	affine_mat=np.matrix(affine)
	return affine_mat







