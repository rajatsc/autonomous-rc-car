from __future__ import print_function
from __future__ import division

import os
import rospy 
import numpy as np
import KinematicModel as model
import cv2
import yaml
from nav_msgs.srv import GetMap
from scipy import signal
import time
import sys
import matplotlib.pyplot as plt

class MapProcessor(object):

    def __init__(self, map_img, red_points, yaml_input, path):
        
        self.red_points = red_points
        self.mapImageGS = map_img
        self.path=path+"/maps/"
        print(self.path)
        self.yaml_input=yaml_input

        #get resolution
        self.map_resolution=yaml_input["resolution"]
        print(self.map_resolution)

        print(np.unique(self.mapImageGS)) #Prints out unique elements in pgm
        
        #get the height and width of the image
        height, width= self.mapImageGS.shape
        self.mapHeight = height
        self.mapWidth = width
    
        #Create a Black and White map
        self.mapImageBW=np.zeros((self.mapHeight, self.mapWidth))
        white_threshold=253
        self.mapImageBW[self.mapImageGS >= white_threshold ] = 255 
        self.mapImageBW[self.mapImageGS  < white_threshold] = 0 
        #print(self.mapImageBW)
    
        #cv2.imwrite('mapImageBW.pgm',self.mapImageBW)

        #Radius of the red_dots
        self.rad=0.23

    def clean_map(self):
        kernel=np.ones((5,5), np.uint8)
        #erosion=cv2.erode(img, kernel, iterations=1)

        self.mapImageBW = cv2.morphologyEx(self.mapImageBW, cv2.MORPH_CLOSE, kernel)
        #cv2.imwrite('clean.png', self.mapImageBW)
        

    # return map with no expanded walls, and (realistic size) circles around each red point
    def get_map_for_planning(self):
        final_pgm_file=os.path.join(self.path, "map_for_planning.pgm")
        final_yaml_file=os.path.join(self.path, "map_for_planning.yaml")
        print(final_yaml_file)

        RADIUS_METERS=self.rad
        radius_pixels = int(RADIUS_METERS / self.map_resolution + 0.5)
        print(radius_pixels)

        result_map = self.mapImageBW.copy()
        for red_point in self.red_points:
            center_pixels=tuple(red_point)
            cv2.circle(result_map, center=center_pixels, radius=radius_pixels, color=0, thickness=-1)

        print(result_map)
        cv2.imwrite(final_pgm_file , img=result_map)

        #Change YAML parameters and save 
        self.yaml_input["image"]='map_for_planning.pgm'
       
        with open(final_yaml_file, "w") as outfile:
            yaml.dump(self.yaml_input, outfile)


    # return map with expanded walls and expanded points
    def get_map_for_mppi(self):

        final_pgm_file=os.path.join(self.path, "map_for_mppi.pgm")
        final_yaml_file=os.path.join(self.path, "map_for_mppi.yaml")

        result_map = self.mapImageBW.copy()

        #RADIUS_METERS=self.rad+0.35
        RADIUS_METERS=self.rad+0.45
        radius_pixels = int(RADIUS_METERS / self.map_resolution + 0.5)
        print(radius_pixels)

        result_map = self.mapImageBW.copy()

        #kernel=np.ones((19,19), np.uint8)
        kernel=np.ones((15,15), np.uint8)
        result_map=cv2.erode(result_map, kernel, iterations=1)
        for red_point in self.red_points:
            center_pixels=tuple(red_point)
            cv2.circle(result_map, center=center_pixels, radius=radius_pixels, color=0, thickness=-1)
        
        print(result_map)
    
        cv2.imwrite(final_pgm_file, img=result_map)
        occu_grid=~result_map.astype(dtype=bool)
        arr_path=os.path.join(self.path, "permissible_region")
        np.save(arr_path, occu_grid)
        #Change YAML parameters and save 
        self.yaml_input["image"]='map_for_mppi.pgm'
        
        with open(final_yaml_file, "w") as outfile:
            yaml.dump(self.yaml_input, outfile)





if __name__=="__main__":

    """
    #getting the static_map service
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    mapFile = rospy.ServiceProxy(map_service_name, GetMap)().map
    """
    #fname='maps/sieg_floor3'
    fname='final/gates'

    #path = '/home/rajat/p0_catkin_ws/src/lab4/cse490-lab4'
    path='/home/nvidia/catkin_ws/src/lab4' 
    red_pt= os.path.join(os.path.join(path, 'final/bad_waypoints.csv'))

    pgm_path = os.path.join(path, fname + '.pgm')
    yaml_path = os.path.join(path, fname+ '.yaml')
    print(pgm_path )
    print(yaml_path)
    
    #Reading .pgm file
    pgm_input=cv2.imread(pgm_path, 0)
    print(pgm_input)
    print(pgm_input.shape)
    #cv2.imshow('image',a)
    #cv2.waitKey(0)

    #cv2.imwrite('myfile.pgm', pgm_input)

    #Reading .yaml file
    with open(yaml_path) as infile:
        yaml_input = yaml.load(infile)

    #print(yaml_input)
    
    #getting red points
    my_data = np.loadtxt(red_pt,dtype=np.int, delimiter=',', skiprows=1, ndmin=2)
    print(my_data)

    my_MapProcessor=MapProcessor(pgm_input, my_data, yaml_input, path)

    print('Start')
    my_MapProcessor.clean_map()
    my_MapProcessor.get_map_for_planning()
    print('Map for Planning done')
    my_MapProcessor.get_map_for_mppi()
    print('Map for MPPI done')
