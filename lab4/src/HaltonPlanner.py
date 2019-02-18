import math
import numpy
import Dubins
import KinematicModel as model
from matplotlib import pyplot as plt
import cv2
import Utils
import time
import random

class HaltonPlanner(object):
  
  # planningEnv: Should be a HaltonEnvironment
  def __init__(self, planningEnv):
    self.planningEnv = planningEnv

  # Generate a plan
  # Assumes that the source and target were inserted just prior to calling this
  # Returns the generated plan
  def plan(self):
    print "Searching..."
    self.sid = self.planningEnv.graph.number_of_nodes() - 2 # Get source id
    self.tid = self.planningEnv.graph.number_of_nodes() - 1 # Get target id

    self.closed = set() # The closed set
    self.parent = {self.sid:None} # A dictionary mapping children to their parents
    #self.open = {self.sid: 0 + self.planningEnv.get_heuristic(self.sid, self.tid)} # The open list
    self.open = Utils.PriorityQueue()
    self.fringe = set()
    self.open.push(self.sid, 0 + self.planningEnv.get_heuristic(self.sid, self.tid))
    self.fringe.add(self.sid)
    self.gValues = {self.sid:0} # A mapping from node to shortest found path length to that node 

    # ------------------------------------------------------------
    # YOUR CODE HERE
    # 
    # Implement A*
    # Functions that you will probably use
    # - self.get_solution()
    # - self.planningEnv.get_successors()
    # - self.planningEnv.get_distance()
    # - self.planningEnv.get_heuristic()
    # Note that each node in the graph has both an associated id and configuration
    # You should be searching over ids, not configurations. get_successors() will return
    #   the ids of nodes that can be reached using an unobstructed Dubins path. Once you have a path plan
    #   of node ids, get_solution() will compute the actual path in SE(2) based off of
    #   the node ids that you have found.
    #-------------------------------------------------------------
    while not self.open.isEmpty():
      current = self.open.pop()
      self.fringe.remove(current)
      
      if current == self.tid:
        print "Search complete!"
        return self.get_solution(current)
      
      self.closed.add(current)
      if len(self.closed) % 10 == 0:
        print "Visited {} nodes.".format(len(self.closed))
      successors = self.planningEnv.get_successors(current)
      for successor in successors:
        if successor in self.closed:
          continue
        dist = self.planningEnv.get_distance(current, successor)
        heur = self.planningEnv.get_heuristic(successor, self.tid)

        tentative_gScore = self.gValues[current] + dist
        score =  tentative_gScore + heur
        if successor not in self.fringe:
          self.open.push(successor, score)
          self.fringe.add(successor)

        if successor in self.gValues and tentative_gScore >= self.gValues[successor]:
          continue

        self.gValues[successor] = tentative_gScore
        self.parent[successor] = current

    print "Search complete! No path found."
    return []

  # Try to improve the current plan by repeatedly checking if there is a shorter path between random pairs of points in the path
  def post_process(self, plan, timeout):

    t1 = time.time()
    elapsed = 0
    print ("prev plan length", len(plan))
    while elapsed < timeout: # Keep going until out of time
      # ---------------------------------------------------------
      # YOUR CODE HERE
      
      # Pseudocode
      
      # Pick random id i
      i = random.randint(0, len(plan)-1)
      # Pick random id j
      j = random.randint(0, len(plan)-1)
      # Redraw if i == j
      if i == j:
        continue
      # Switch i and j if i > j
      if i > j:
        temp = i
        i = j
        j = temp
     
      i_config = plan[i]
      j_config = plan[j]
      
      if Utils.euclidian_distance(i_config, j_config) < 1:
        continue
      # if we can find path between i and j (Hint: look inside HaltonEnvironment.py for a suitable function)
      if self.planningEnv.manager.get_edge_validity(i_config, j_config):
        # Get the path (Hint: use Dubins) JOHAN: but we don't really need to compute the path
        px, py, pyaw, clen = Dubins.dubins_path_planning(i_config, j_config, 1.0/model.TURNING_RADIUS)
        # Reformat the plan such that the new path is inserted and the old section of the path is removed between i and j
        # TODO check edge cases
        plan_start = plan[0:i]
        plan_end   = plan[j:]
        plan = plan_start + list(zip(px, py, pyaw)) + plan_end
          
        # Be sure to CAREFULLY inspect the data formats of both the original plan and the plan returned by Dubins 
        # to ensure that you edit the path correctly
      # ----------------------------------------------------------
      elapsed = time.time() - t1
    print ("next plan length", len(plan))
    return plan

  # Backtrack across parents in order to recover path
  # vid: The id of the last node in the graph
  def get_solution(self, vid):

    # Get all the node ids
    planID = [] 
    while vid is not None:
      planID.append(vid)
      vid = self.parent[vid]

    plan = []
    planID.reverse() # Fix backwards ids
    for i in range(len(planID)-1): # Find path from one id to the next
      startConfig = self.planningEnv.get_config(planID[i])
      goalConfig = self.planningEnv.get_config(planID[i+1])
      px, py, pyaw, clen = Dubins.dubins_path_planning(startConfig, goalConfig, 1.0/model.TURNING_RADIUS)
      plan.append([list(a) for a in zip(px,py,pyaw)])

    flatPlan = [item for sublist in plan for item in sublist]
    return flatPlan

  # Visualize the plan
  def simulate(self, plan, name=None):
    # Get the map
    envMap = 255*(self.planningEnv.manager.mapImageBW+1) # Hacky way to get correct coloring
    #envMap = cv2.cvtColor(envMap, cv2.COLOR_GRAY2RGB)
    
    for i in range(numpy.shape(plan)[0]-1): # Draw lines between each configuration in the plan
      startPixel = Utils.our_world_to_map(plan[i], self.planningEnv.manager.map_info)
      goalPixel = Utils.our_world_to_map(plan[i+1], self.planningEnv.manager.map_info)
      cv2.line(envMap,(int(startPixel[0]),int(startPixel[1])),(int(goalPixel[0]),int(goalPixel[1])),0,5)

    # Generate window
    #cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
    #envMap = cv2.cvtColor(envMap, cv2.COLOR_GRAY2RGB)
    if name is None:
      cv2.imwrite('Simulation.png', envMap)
    else:
      cv2.imwrite(name, envMap)
    
