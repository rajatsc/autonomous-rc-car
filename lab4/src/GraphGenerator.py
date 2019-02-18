#!/usr/bin/env python

# Import standard python libraries
import rospy
from nav_msgs.srv import GetMap
import networkx as nx
import math
import numpy
import numpy as np
import Dubins
import KinematicModel as model
from ObstacleManager import ObstacleManager
import rospy
from nav_msgs.srv import GetMap
import matplotlib.pyplot as plt
import Utils

np.random.seed(0)
numpy.random.seed(0)

# Halton Sequence Generator
def halton_sequence_value(index, base):

    result = 0
    f = 1

    while index > 0:
        f = f*1.0/base
        result = result + f*(index % base)
        index = index/base

    return result

# Wrap the values around 0 and 1
def wrap_around(coordinate):

    for i in range(numpy.size(coordinate)):
        if coordinate[i] > 1.0:
            coordinate[i] = coordinate[i] - 1.0
        if coordinate[i] < 0:
            coordinate[i] = 1.0 + coordinate[i]

    return coordinate

# Halton Graph Generator
def euclidean_halton_graph(n, radius, bases, lower, upper, source, target, mapFile):

    manager = ObstacleManager(mapFile)

    G = nx.DiGraph()
    upper = numpy.array(upper)
    lower = numpy.array(lower)
    scale = upper-lower
    offset = lower

    print("scale", scale)

    position = []

    numVertices = 0
    haltonIndex = 1

    if source is not None:
      print("source", source)
      position.append(source)
      numVertices += 1
    if target is not None:
      print("target", target)
      position.append(target)
      numVertices += 1

    print('num vertices', numVertices)
    print('n', n)
    while numVertices < n:
        p = wrap_around(numpy.array([halton_sequence_value(haltonIndex,base) for base in bases]))
        p = p * scale + offset

        if manager.get_state_validity(p):
            position.append(p)
            numVertices += 1

        haltonIndex += 1
    print("position", position)
    print("position", [type(p) for p in position])
    state = [" ".join(str(x) for x in p) for p in position]

    for i in range(n):
        node_id = i
        G.add_node(str(node_id), state = state[i])

    for i in range(n-1):
        print i
        for j in range(i+1,n):
            edgeLength = Dubins.path_length(position[i], position[j], 1.0/model.TURNING_RADIUS)
            euclideanLength = numpy.linalg.norm(position[i][0:2] - position[j][0:2])
            if edgeLength < radius:
                G.add_edge(str(i), str(j), length = str(edgeLength))
            edgeLength = Dubins.path_length(position[j], position[i], 1.0/model.TURNING_RADIUS)
            if edgeLength < radius:
                G.add_edge(str(j), str(i), length = str(edgeLength))
    pos = {}
    for n in list(G.nodes(data = True)):
      pose = numpy.array(map(float, n[1]['state'].split(' ')))
      pxl_pose = Utils.our_world_to_map(pose, mapFile.info)
      pos[n[0]] = (pxl_pose[0], pxl_pose[1])
    return G, pos

def euclidean_uniform_graph(n, radius, bases, lower, upper, source, target, mapFile):

    manager = ObstacleManager(mapFile)

    G = nx.DiGraph()
    upper = numpy.array(upper)
    lower = numpy.array(lower)
    scale = upper-lower
    offset = lower

    print("scale", scale)

    position = []

    numVertices = 0

    if source is not None:
      print("source", source)
      position.append(source)
      numVertices += 1
    if target is not None:
      print("target", target)
      position.append(target)
      numVertices += 1

    print("SHAPE",manager.mapImageBW.shape)
    valid_xs, valid_ys, _ = np.where(manager.mapImageBW == 0)
    random_indices = np.arange(valid_xs.shape[0])
    numpy.random.shuffle(random_indices)
    selected_xs, selected_ys = valid_xs[random_indices], valid_ys[random_indices]
    selected_thetas = np.random.uniform(0, 2*np.pi, valid_xs.shape[0])

    print('num vertices', numVertices)
    print('n', n)
    selected_configs = np.column_stack((selected_xs, selected_ys, selected_thetas))
    Utils.map_to_world(selected_configs, mapFile.info)
    for p in selected_configs:

        if manager.get_state_validity(p):
            position.append(p)
            numVertices += 1

        if numVertices >= n:
            break


    print("position", position)
    print("position", [type(p) for p in position])
    state = [" ".join(str(x) for x in p) for p in position]

    for i in range(n):
        node_id = i
        G.add_node(str(node_id), state = state[i])

    for i in range(n-1):
        print i
        for j in range(i+1,n):
            edgeLength = Dubins.path_length(position[i], position[j], 1.0/model.TURNING_RADIUS)
            euclideanLength = numpy.linalg.norm(position[i][0:2] - position[j][0:2])
            if edgeLength < radius:
                G.add_edge(str(i), str(j), length = str(edgeLength))
            edgeLength = Dubins.path_length(position[j], position[i], 1.0/model.TURNING_RADIUS)
            if edgeLength < radius:
                G.add_edge(str(j), str(i), length = str(edgeLength))
    pos = {}
    for n in list(G.nodes(data = True)):
      pose = numpy.array(map(float, n[1]['state'].split(' ')))
      pxl_pose = Utils.our_world_to_map(pose, mapFile.info)
      pos[n[0]] = (pxl_pose[0], pxl_pose[1])
    return G, pos

def insert_vertices(G, configs, radius):

    numVertices = G.number_of_nodes()
    for config in configs:
        state = " ".join(str(x) for x in config)
        G.add_node(str(numVertices), state = state)
        for i in range(numVertices):
            position = [float(a) for a in G.node[str(i)]["state"].split()]

            edgeLength = Dubins.path_length(config, position, 1.0/model.TURNING_RADIUS)
            if edgeLength < radius:
                G.add_edge(str(numVertices), str(i), length = str(edgeLength))

            edgeLength = Dubins.path_length(position, config, 1.0/model.TURNING_RADIUS)
            if edgeLength < radius:
                G.add_edge(str(i), str(numVertices), length = str(edgeLength))
        numVertices += 1

    #nx.write_graphml(G, "currentHalton.graphml")

# Main Function
if __name__ == "__main__":
    riskmapFile = 'gates_dense_obstacles_mppi_final.graphml'

    rospy.init_node("generate_graph")
    
    spaceDimension = 3

    if spaceDimension == 3:
        bases = [2,3,5]

    # Get the map
    map_service_name = rospy.get_param("~static_map", "/mppi/static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    mapFile = rospy.ServiceProxy(map_service_name, GetMap)().map
    map_info = mapFile.info

    w = mapFile.info.width
    h = mapFile.info.height
    print(w,h)
    lower = numpy.array([-h,-w]) * mapFile.info.resolution
    upper = numpy.array([h,w]) * mapFile.info.resolution
    lower = numpy.append(lower, 0.0)
    upper = numpy.append(upper, 2*numpy.pi)
    #lower = numpy.array([map_info.origin.position.x, map_info.origin.position.y,0.0])
    #upper = numpy.array([map_info.origin.position.x+map_info.resolution*map_info.width, map_info.origin.position.y+map_info.resolution*map_info.height, 2*numpy.pi])
    #lower = numpy.append(numpy.zeros(spaceDimension-1), 0.0)
    #upper = numpy.append(numpy.ones(spacedimension-1), 2*numpy.pi)
    print("lower", lower)
    print("upper", upper)


    # Settings
    halton_points = 300
    disc_radius = 6 * halton_points**(-0.5) / mapFile.info.resolution #5*halton_points**(-0.5) 

    print(disc_radius)


    for i in range(1):
        print i
        numpy.random.seed()
        offset = numpy.random.random_sample(spaceDimension,)

        # Generate the graph
        print("upper", upper)
        print("lower", lower)
        print 'Generating the graph'
        G, pos = euclidean_halton_graph(halton_points, disc_radius, bases, lower, upper, offset, None, mapFile)
        #G, pos = euclidean_uniform_graph(halton_points, disc_radius, bases, lower, upper, offset, None, mapFile)
        print(pos)
        nx.write_graphml(G, riskmapFile)
        print("graph saved as {}".format(riskmapFile))
        nx.draw(G, pos=pos)
        plt.draw()
        plt.show()
