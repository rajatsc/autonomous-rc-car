import cv2
import math
import numpy
import Dubins
import Utils
import KinematicModel as model

class ObstacleManager(object):

  def __init__(self, mapMsg):
    # Setup the map
    self.map_info = mapMsg.info
    self.mapImageGS = numpy.array(mapMsg.data, dtype=numpy.uint8).reshape((mapMsg.info.height, mapMsg.info.width,1))

    # Retrieve the map dimensions
    height, width, channels = self.mapImageGS.shape
    self.mapHeight = height
    self.mapWidth = width
    self.mapChannels = channels

    # Binarize the Image
    self.mapImageBW = 255*numpy.ones_like(self.mapImageGS, dtype=numpy.uint8)
    self.mapImageBW[self.mapImageGS==0] = 0
    #self.mapImageBW = self.mapImageBW[::-1,:,:] # Need to flip across the y-axis    
    
    # Obtain the car length and width in pixels
    self.robotWidth = int(model.CAR_WIDTH/self.map_info.resolution + 0.5)
    self.robotLength = int(model.CAR_LENGTH/self.map_info.resolution + 0.5)

  # Check if the passed config is in collision
  # config: The configuration to check (in meters and radians)
  # Returns False if in collision, True if not in collision
  def get_state_validity(self, config, in_world_frame=True):

    # Convert the configuration to map-coordinates -> mapConfig is in pixel-space

    if in_world_frame: 
      mapConfig = Utils.our_world_to_map(config, self.map_info)
    else:
      mapConfig = config

    # ---------------------------------------------------------
    # YOUR CODE HERE
    #
    # Return true or false based on whether the configuration is in collision
    # Use self.robotWidth and self.robotLength to represent the size of the robot
    # Also return false if the robot is out of bounds of the map
    # Although our configuration includes rotation, assume that the
    # rectangle representing the robot is always aligned with the coordinate axes of the
    # map for simplicity
    # ----------------------------------------------------------
    half_width = int(self.robotWidth / 2)
    half_length = int(self.robotLength / 2)

    center = (mapConfig[1], mapConfig[0])
    left   = (mapConfig[1], mapConfig[0] - half_width)
    right  = (mapConfig[1], mapConfig[0] + half_width)
    top    = (mapConfig[1] + half_length, mapConfig[0])             
    bot    = (mapConfig[1] - half_length, mapConfig[0])
    topL   = (mapConfig[1] + half_length, mapConfig[0] - half_width)
    topR   = (mapConfig[1] + half_length, mapConfig[0] + half_width)
    botL   = (mapConfig[1] - half_length, mapConfig[0] - half_width)
    botR   = (mapConfig[1] - half_length, mapConfig[0] + half_width)

    points = [center, top, bot, left, right, topL, topR, botL, botR]
 
    for point in points:
      if point[0] < 0 or point[1] < 0:
        return False
      if point[1] >= self.mapWidth or point[0] >= self.mapHeight:
        return False
      if self.mapImageBW[point] != 0:
        return False
    return True  

  # Check if there is an unobstructed edge between the passed configs
  # config1, config2: The configurations to check (in meters and radians)
  # Returns false if obstructed edge, True otherwise
  def get_edge_validity(self, config1, config2):
    # -----------------------------------------------------------
    # YOUR CODE HERE
    #
    # Check if endpoints are obstructed, if either is, return false
    # Find path between two configs using Dubins
    # Check if all configurations along Dubins path are obstructed 
    # -----------------------------------------------------------
    if not self.get_state_validity(config1) or not self.get_state_validity(config2):
      return False

    px, py, pyaw, cost = Dubins.dubins_path_planning(config1, config2, 1.0/model.TURNING_RADIUS)
    dubins_path = zip(px, py, pyaw)

    for config in dubins_path:
      if not self.get_state_validity(config):
        return False

    return True


# Test
if __name__ == '__main__':
  pass
  # Write test code here!
