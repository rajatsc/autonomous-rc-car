#!/usr/bin/env python


from __future__ import division
from __future__ import print_function

import time
import sys
import os
import rospy
import rosbag
import numpy as np
import scipy.signal
import utils as Utils
import pickle
import matplotlib.pyplot as plt
from scipy import stats

try:
  import torch
  import torch.utils.data
  from torch.autograd import Variable
  import torch.nn.functional as F
except ImportError:
  print('No torch module found')


##################################################
INPUT_SIZE=8
OUTPUT_SIZE=3
DATA_SIZE=6
L=0.33 #Length of the car


# TODO
# specify your neural network (or other) model here.
# model = torch
def gen_nn():
  n_feature = INPUT_SIZE
  n_output = OUTPUT_SIZE

  n_hidden1 = 20
  n_hidden2 = 20
  n_hidden3=20

  """
  List of Activation functions

  ReLU
  Sigmoid
  Tanh
  Threshold
  SELU
  GLU

  """

  layers = []
  layers.append(torch.nn.Linear(n_feature, n_hidden1))
  layers.append(torch.nn.Tanh())
  layers.append(torch.nn.Linear(n_hidden1, n_hidden2))
  layers.append(torch.nn.Tanh())
  layers.append(torch.nn.Linear(n_hidden2, n_hidden3))
  layers.append(torch.nn.Tanh())
  layers.append(torch.nn.Linear(n_hidden3, n_output))
  
  return torch.nn.Sequential(*layers)

def do_training(model, filename, optimizer, x, y, x_val, y_val, loss_fn, N=10000):
    for t in range(N):
        y_pred = model(Variable(x))
        loss = loss_fn(y_pred, Variable(y, requires_grad=False))
        if t % 50 == 0:
            val = model(Variable(x_val))
            vloss = loss_fn(val, Variable(y_val, requires_grad=False))
            print(t, loss.data[0]/x.shape[0], vloss.data[0]/x_val.shape[0])

            ##final_loss

        optimizer.zero_grad() # clear out old computed gradients
        loss.backward()       # apply the loss function backprop
        optimizer.step()      # take a gradient step for model's parameters
    
    #print('Final Validation set error')
    #print(val_pred)
    val_pred=torch.mean(torch.abs(val.data-y_val), 0, keepdim=True)
    print(val_pred.cpu().numpy())

    torch.save(model, filename)

# The following are functions meant for debugging and sanity checking your
# model. You should use these and / or design your own testing tools.
# test_model starts at [0,0,0]; you can specify a control to be applied and the
# rollout() function will use that control for N timesteps.
# i.e. a velocity value of 0.7 should drive the car to a positive x value.

def rollout(m, nn_input, N, learn_residuals):
    if learn_residuals==True:
        print('Generate Kinematic+NN rollouts')
        pose=torch.zeros(3).cuda()
        output_pose=np.zeros((10,3))
        out_km_tensor=torch.zeros(3).cuda()
        print (pose.cpu().numpy())
        for i in range(N):
            #convert nn_input to numpy array
            temp_nn_input=nn_input
            np_nn_input=temp_nn_input.cpu().numpy()
            #print(np_nn_input.shape)
            #extract steering controls
            speed_km=np_nn_input[5]
            steer_km=np_nn_input[6]
            sine_km=np_nn_input[3]
            cosine_km=np_nn_input[4]
            dt_km=np_nn_input[7]
            sin_2_beta=np.sin(2.0*np.arctan(np.tan(steer_km)/2.0))

            """
            print(speed_km)
            print(steer_km)
            print(sine_km)
            print(cosine_km)
            print(dt_km)
            print(sin_2_beta)
            """

            np_pose=pose.cpu().numpy()
            old_orientation=np_pose[2]
            #print('Printing old orientation')
            #print(old_orientation)
            out_km=Ackerman_equations_rollouts(speed_km, steer_km, sine_km, cosine_km, dt_km, sin_2_beta, np.asscalar(old_orientation))
            
            out_km_tensor[0]=out_km[0]
            out_km_tensor[1]=out_km[1]
            out_km_tensor[2]=out_km[2]


            out_res=m(Variable(nn_input))
            #print(out_res.data)
            #print(out_km_tensor)

            out=out_km_tensor-out_res.data
            #print('yo')
            pose=pose+out
            #Wrap pi
            if pose[2] > 3.14:
                pose[2] -= 2*np.pi

            if pose[2] < -3.14:
                pose[2] += 2*np.pi
            nn_input[0]=out[0]
            nn_input[1]=out[1]
            nn_input[2]=out[2]
            nn_input[3]=np.sin(pose[2])
            nn_input[4]=np.cos(pose[2])
            print(pose.cpu().numpy())
            output_pose[i,:]=pose.cpu().numpy()
        return(output_pose)
    else:

      pose = torch.zeros(3).cuda()
      output_pose=np.zeros((10,3))
      print(pose.cpu().numpy())
      for i in range(N):
          out = m(Variable(nn_input))
          #print(out)
          pose.add_(out.data)
          # Wrap pi
          if pose[2] > 3.14:
              pose[2] -= 2*np.pi

          if pose[2] < -3.14:
              pose[2] += 2*np.pi
          nn_input[0] = out.data[0]
          nn_input[1] = out.data[1]
          nn_input[2] = out.data[2]
          nn_input[3] = np.sin(pose[2])
          nn_input[4] = np.cos(pose[2])
          print(pose.cpu().numpy())
          output_pose[i,:]=pose.cpu().numpy()
      return(output_pose)
     
def test_model(m, N, learn_residuals, dt = 0.1):


    TEST_CASES=5
    output_pose=np.zeros((5,10,3))

    cos, v, st = 4, 5, 6

    s = INPUT_SIZE 
    print("Nothing")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[7] = dt
    out=rollout(m, nn_input, N, learn_residuals)
    output_pose[0,:,:]=out

    print("Forward")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = 0.7 #1.0
    nn_input[7] = dt
    out=rollout(m, nn_input, N, learn_residuals)
    output_pose[1,:,:]=out

    print("Backward")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = - 0.7 #1.0
    nn_input[7] = dt
    out=rollout(m, nn_input, N, learn_residuals)
    output_pose[2,:,:]=out

    print("Forward and Steer Right")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = 0.7 #1.0
    nn_input[st]= 0.34
    nn_input[7] = dt
    out=rollout(m, nn_input, N, learn_residuals)
    output_pose[3,:,:]=out

    print("Forward and Steer Left")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] =  0.7 #1.0
    nn_input[st]= - 0.34
    nn_input[7] = dt
    out=rollout(m, nn_input, N, learn_residuals)
    output_pose[4,:,:]=out

    if learn_residuals:
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/nothing_rl.txt', output_pose[0,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/forward_rl.txt', output_pose[1,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/backward_rl.txt', output_pose[2,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/forward_rightsteer_rl.txt', output_pose[3,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/forward_leftsteer.txt_rl', output_pose[4,:,:])
    else:

        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/nothing.txt', output_pose[0,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/forward.txt', output_pose[1,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/backward.txt', output_pose[2,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/forward_rightsteer.txt', output_pose[3,:,:])
        np.savetxt('/home/nvidia/catkin_ws/src/lab3/txt_files/forward_leftsteer.txt', output_pose[4,:,:])
        print("rollouts saved!")




def threshold_data(data, threshold_array):
    data=np.asarray(data)
    absolute_data=np.abs(data)
    rows,cols=data.shape
    for i in range(cols):
      low_value_flags=absolute_data[:,i] < threshold_array[i]
      data[low_value_flags,i]=0

    return data


def extract_data(raw_datas):
  x_datas = np.zeros( (raw_datas.shape[0]-2, INPUT_SIZE) )
  y_datas = np.zeros( (raw_datas.shape[0]-2, OUTPUT_SIZE) )
  pose_dot = np.zeros( (raw_datas.shape[0]-1, OUTPUT_SIZE) )

  # calculate pose diff
  pose_dot[:,0] = np.diff(raw_datas[:,0]) #x_dot
  pose_dot[:,1] = np.diff(raw_datas[:,1]) #y_dot
  pose_dot[:,2] = np.diff(raw_datas[:,2]) #theta_dot

  # TODO
  # It is critical we properly handle theta-rollover: 
  # as -pi < theta < pi, theta_dot can be > pi, so we have to handle those
  # cases to keep theta_dot also between -pi and pi

  # Handle positive case
  dt = np.diff(raw_datas[:,5])
  dt = dt.reshape((dt.shape[0],1))
  
  gt = pose_dot[:,2] > np.pi
  pose_dot[gt,2] = pose_dot[gt,2] - 2*np.pi
  # Handle negative case
  gt = pose_dot[:,2] <  -np.pi
  pose_dot[gt,2] = pose_dot[gt,2] + 2*np.pi

  # Scale by dt
  #pose_dot[:,:] *= 1.0/dt * .1

  x_datas[:,0] = pose_dot[:-1,0] #x_dot
  x_datas[:,1] = pose_dot[:-1,1] #y_dot
  x_datas[:,2] = pose_dot[:-1,2] #theta_dot
  theta = raw_datas[1:-1,2]
  x_datas[:,3] = np.sin(theta) #sin(theta)
  x_datas[:,4] = np.cos(theta) #cos(theta)
  x_datas[:,5] = raw_datas[1:-1,3] #v
  x_datas[:,6] = raw_datas[1:-1,4] #theta
  x_datas[:,7] = np.diff(raw_datas[:-1,5]) #dt

  y_datas[:,0] = pose_dot[1:,0] #x_dot
  y_datas[:,1] = pose_dot[1:,1] #y_dot
  y_datas[:,2] = pose_dot[1:,2] #theta_dot
  return (x_datas, y_datas)

def Ackerman_equations_rollouts(speed_km, steer_km, sine_km, cosine_km, dt_km, sin_2_beta, old_orientation):

    delta_pose_km=np.zeros(3)
    if steer_km==0:
        delta_pose_km[0]=speed_km * cosine_km * dt_km
        delta_pose_km[1]=speed_km * sine_km* dt_km
        delta_pose_km[2]=0
    else:
        delta_pose_km[2]=(speed_km/L)*\
                           sin_2_beta * dt_km
                   
        sine_one_km=np.sin(old_orientation+delta_pose_km[2])
        cosine_one_km=np.cos(old_orientation+delta_pose_km[2])
        #print(sine_one_km)
        delta_pose_km[0]=(L / sin_2_beta) * \
                       (sine_one_km - sine_km)
        delta_pose_km[1]=(L / sin_2_beta) * \
                        (-cosine_one_km + cosine_km)


    return delta_pose_km


def Ackerman_equations_1(delta_pose_km, speed_km, steer_km, sine_km, cosine_km, dt_km, sine_one_km, sin_2_beta):

    zerosteer_flags = np.abs(steer_km)<=0
    print(zerosteer_flags)
    if zerosteer_flags.size:
        print(zerosteer_flags.shape[0])
    print(np.count_nonzero(zerosteer_flags))

    #Ackerman equations for zero steering angle
    delta_pose_km[zerosteer_flags,0]=speed_km[zerosteer_flags] * cosine_km[zerosteer_flags] * dt_km[zerosteer_flags]
    delta_pose_km[zerosteer_flags,1]=speed_km[zerosteer_flags] * sine_km[zerosteer_flags] * dt_km[zerosteer_flags]
    delta_pose_km[zerosteer_flags,2]=0

        #Ackerman equations for non-zero steering angle
            
    delta_pose_km[~zerosteer_flags,0]=(L / sin_2_beta[~zerosteer_flags]) * \
                           (sine_one_km[~zerosteer_flags] - sine_km[~zerosteer_flags])
    delta_pose_km[~zerosteer_flags,1]=(L / sin_2_beta[~zerosteer_flags]) * \
                            (-cosine_one_km[~zerosteer_flags] + cosine_km[~zerosteer_flags])
    delta_pose_km[~zerosteer_flags,2]=(speed_km[~zerosteer_flags]/L)*\
                               sin_2_beta[~zerosteer_flags] * dt_km[~zerosteer_flags]


    return delta_pose_km

if __name__=="__main__":
    
    FILTER = True
    NN = True
    LEARN_RESIDUALS = not FILTER
    THRESHOLD=False
    print(FILTER)
    print(LEARN_RESIDUALS)

    SPEED_TO_ERPM_OFFSET     = 0.0
    SPEED_TO_ERPM_GAIN       = 4614.0
    STEERING_TO_SERVO_OFFSET = 0.5304
    STEERING_TO_SERVO_GAIN   = -1.2135

    if len(sys.argv) < 2:
        print('Input a bag file from command line')
        print('Input a bag file from command line')
        print('Input a bag file from command line')
        print('Input a bag file from command line')
        print('Input a bag file from command line')
        sys.exit()
    base, ext = os.path.splitext(sys.argv[1])
    if ext == '.pickle':
      print('Loading from pickle!')
      raw_datas = pickle.load(open(sys.argv[1], 'rb'))
    elif ext == '.bag':
      print('Loading from bag!')
      bag = rosbag.Bag(sys.argv[1])
      tandt = bag.get_type_and_topic_info()
      t1='/vesc/sensors/core'
      t2='/vesc/sensors/servo_position_command'
      t3='/pf/ta/viz/inferred_pose'
      topics = [t1,t2,t3]
      min_datas = tandt[1][t3][1] # number of t3 messages is less than t1, t2

      raw_datas = np.zeros((min_datas,DATA_SIZE))

      last_servo, last_vel = 0.0, 0.0
      n_servo, n_vel = 0, 0
      idx=0
      # The following for-loop cycles through the bag file and averages all control
      # inputs until an inferred_pose from the particle filter is recieved. We then
      # save that data into a buffer for later processing.
      # You should experiment with additional data streams to see if your model
      # performance improves.
      for topic, msg, t in bag.read_messages(topics=topics):
          if topic == t1:
              last_vel   += (msg.state.speed - SPEED_TO_ERPM_OFFSET) / SPEED_TO_ERPM_GAIN
              n_vel += 1
          elif topic == t2:
              last_servo += (msg.data - STEERING_TO_SERVO_OFFSET) / STEERING_TO_SERVO_GAIN
              n_servo += 1
          elif topic == t3 and n_vel > 0 and n_servo > 0:
              timenow = msg.header.stamp
              last_t = timenow.to_sec()
              last_vel /= n_vel
              last_servo /= n_servo
              orientation = Utils.quaternion_to_angle(msg.pose.orientation)
              data = np.array([msg.pose.position.x,
                               msg.pose.position.y,
                               orientation,
                               last_vel,
                               last_servo,
                               last_t])
              raw_datas[idx,:] = data
              last_vel = 0.0
              last_servo = 0.0
              n_vel = 0
              n_servo = 0
              idx = idx+1
              if idx % 1000==0:
                  print('.')
      bag.close()
      raw_datas = raw_datas[:idx, :] # Clip to only data found from bag file
      
      if len(sys.argv) > 2:
        if sys.argv[2] == 'save':
          pickle.dump(raw_datas, open(sys.argv[3], 'wb'))
          print("Pickle saved to " + sys.argv[3] + "!")
        else:
          print('Usage: bag_file_name save pickle_file_name')
    else:
      print('Extension not recognized!')
    
        

    # Pre-process the data to remove outliers, filter for smoothness, and calculate
    # values not directly measured by sensors

    # Note:
    # Neural networks and other machine learning methods would prefer terms to be
    # equally weighted, or in approximately the same range of values. Here, we can
    # keep the range of values to be between -1 and 1, but any data manipulation we
    # do here from raw values to our model input we will also need to do in our
    # MPPI code.

    # We have collected:
    # raw_datas = [ x, y, theta, v, delta, dt]
    # We want to have:
    # x_datas[i-1,:] = [x_dot, y_dot, theta_dot, sin(theta), cos(theta), v, delta, dt]
    # y_datas[i,  :] = [x_dot, y_dot, theta_dot ]

    #raw_datas = raw_datas[ np.abs(raw_datas[:,3]) < 0.75 ] # discard bad controls
    #raw_datas = raw_datas[ np.abs(raw_datas[:,4]) < 0.36 ] # discard bad controls
    

    x_datas, y_datas = extract_data(raw_datas)

    #####################################################################################################

    if LEARN_RESIDUALS:

        # filter x_dot
        x_datas[:,0] = scipy.signal.savgol_filter(x_datas[:,0], 5, 3)
        y_datas[:,0] = scipy.signal.savgol_filter(y_datas[:,0], 5, 3)

        # filter y_dot
        x_datas[:,1] = scipy.signal.savgol_filter(x_datas[:,1], 5, 3)
        y_datas[:,1] = scipy.signal.savgol_filter(y_datas[:,1], 5, 3)
        # filter theta_dot
        x_datas[:,2] = scipy.signal.savgol_filter(x_datas[:,2], 5, 3)
        y_datas[:,2] = scipy.signal.savgol_filter(y_datas[:,2], 5, 3)
        
        delta_pose_km=np.zeros((y_datas.shape[0]-1,3))

        sine_km = x_datas[:-1,3] 
        sine_one_km = x_datas[1:,3]
        
        cosine_km = x_datas[:-1,4]
        cosine_one_km = x_datas[1:,4]

        speed_km = x_datas[:-1,5]
        steer_km = x_datas[:-1,6]
        dt_km = x_datas[:-1,7]

        sin_2_beta = np.sin(2.0*np.arctan(np.tan(steer_km)/2.0))

        print('Computing residuals')

        delta_pose_km=Ackerman_equations_1(delta_pose_km, speed_km, steer_km, sine_km, cosine_km, dt_km, sine_one_km, sin_2_beta)
        #delta_pose_km_2=Ackerman_equations_2(delta_pose_km,speed_km, steer_km, sine_km, cosine_km, dt_km, sine_one_km, sin_2_beta)
    
        #print(np.sum(delta_pose_km-delta_pose_km_2))

        #Calculate Residuals
        # prune extraneous indices     

        y_datas=delta_pose_km-y_datas[:-1,:]
        x_datas=x_datas[:-1,:]
        print('Computing residuals done')

        outlier_idxs = np.abs(y_datas[:,0]) < 0.1
        x_datas = x_datas[outlier_idxs] 
        y_datas = y_datas[outlier_idxs] 
         
        outlier_idxs = np.abs(y_datas[:,1]) < 0.1
        x_datas = x_datas[outlier_idxs] 
        y_datas = y_datas[outlier_idxs] 

        print('Plotting')

        #plt.plot(np.arange(steer_km.shape[0]), steer_km)
        #plt.plot(np.arange(delta_pose_km[:,0].shape[0]) , x_datas[:,0])
        plt.plot(np.arange(y_datas[:,0].shape[0]) , y_datas[:,0])
        #plt.plot(np.arange(delta_pose_km_2[:,0].shape[0]) , delta_pose_km_2[:,0])
        #plt.plot(np.arange(speed_km.shape[0]), speed_km)
        #plt.plot(np.arange(sin_2_beta.shape[0]), sin_2_beta)
        plt.show()




    v_idxs = np.abs(x_datas[:,5]) < 0.95
    delta_idxs = np.abs(x_datas[:,6]) < 0.36
    accept_flags=np.logical_and(v_idxs, delta_idxs)

    x_datas=x_datas[accept_flags]
    y_datas=y_datas[accept_flags]
    """
    x_datas = x_datas[v_idxs]
    y_datas = y_datas[v_idxs] # discard bad controls
    x_datas = x_datas[delta_idxs]
    y_datas = y_datas[delta_idxs]
    """
    index = 1

    #plt.plot(np.arange(x_datas.shape[0]-50), x_datas[50:,index])
    #plt.plot(np.arange(150), x_datas[50:200,index])  
    if FILTER:
      x_datas_filtered, y_datas_filtered = np.copy(x_datas), np.copy(y_datas)
      # plot original fn
      #plt.plot(np.arange(x_datas.shape[0]-50), x_datas[50:,index])
      # filter x_dot
      x_datas_filtered[:,0] = scipy.signal.savgol_filter(x_datas_filtered[:,0], 5, 3)
      y_datas_filtered[:,0] = scipy.signal.savgol_filter(y_datas_filtered[:,0], 5, 3)

      # filter y_dot
      x_datas_filtered[:,1] = scipy.signal.savgol_filter(x_datas_filtered[:,1], 5, 3)
      y_datas_filtered[:,1] = scipy.signal.savgol_filter(y_datas_filtered[:,1], 5, 3)
      # filter theta_dot
      x_datas_filtered[:,2] = scipy.signal.savgol_filter(x_datas_filtered[:,2], 5, 3)
      y_datas_filtered[:,2] = scipy.signal.savgol_filter(y_datas_filtered[:,2], 5, 3)
      # filter sin(theta)
      #x_datas_filtered[:,3] = scipy.signal.savgol_filter(x_datas_filtered[:,3], 21, 2)
      # filter cos(theta)
      #x_datas_filtered[:,4] = scipy.signal.savgol_filter(x_datas_filtered[:,4], 21, 2)
      # filter v
      #x_datas_filtered[:,5] = scipy.signal.savgol_filter(x_datas_filtered[:,5], 35, 2)
      # filter delta
      #x_datas_filtered[:,6] = scipy.signal.savgol_filter(x_datas_filtered[:,6], 3, 2)
      # filter dt
      #x_datas_filtered[:,7] = scipy.signal.savgol_filter(x_datas_filtered[:,7], 31, 3)

      ######## plot filtered fn
      #plt.plot(np.arange(x_datas_filtered.shape[0]-50), x_datas_filtered[50:,index])
      #plt.plot(np.arange(150), x_datas_filtered[50:200,index])
    # TODO
    # Some raw values from sensors / particle filter may be noisy. It is safe to
    # filter the raw values to make them more well behaved. We recommend something
    # like a Savitzky-Golay filter. You should confirm visually (by plotting) that
    # your chosen smoother works as intended.
    # An example of what this may look like is in the homework document.

    # Convince yourself that input/output values are not strange
    print("Xdot  ", np.min(x_datas[:,0]), np.max(x_datas[:,0]))
    print("Ydot  ", np.min(x_datas[:,1]), np.max(x_datas[:,1]))
    print("Tdot  ", np.min(x_datas[:,2]), np.max(x_datas[:,2]))
    print("sin   ", np.min(x_datas[:,3]), np.max(x_datas[:,3]))
    print("cos   ", np.min(x_datas[:,4]), np.max(x_datas[:,4]))
    print("vel   ", np.min(x_datas[:,5]), np.max(x_datas[:,5]))
    print("delt  ", np.min(x_datas[:,6]), np.max(x_datas[:,6]))
    print("dt    ", np.min(x_datas[:,7]), np.max(x_datas[:,7]))
    print()
    print("y Xdot", np.min(y_datas[:,0]), np.max(y_datas[:,0]))
    print("y Ydot", np.min(y_datas[:,1]), np.max(y_datas[:,1]))
    print("y Tdot", np.min(y_datas[:,2]), np.max(y_datas[:,2]))
    #print("first 50  ", x_datas[1000:1050,:])
    
    if FILTER:
      print()
      print("Xdot  ", np.min(x_datas_filtered[:,0]), np.max(x_datas_filtered[:,0]))
      print("Ydot  ", np.min(x_datas_filtered[:,1]), np.max(x_datas_filtered[:,1]))
      print("Tdot  ", np.min(x_datas_filtered[:,2]), np.max(x_datas_filtered[:,2]))
      print("sin   ", np.min(x_datas_filtered[:,3]), np.max(x_datas_filtered[:,3]))
      print("cos   ", np.min(x_datas_filtered[:,4]), np.max(x_datas_filtered[:,4]))
      print("vel   ", np.min(x_datas_filtered[:,5]), np.max(x_datas_filtered[:,5]))
      print("delt  ", np.min(x_datas_filtered[:,6]), np.max(x_datas_filtered[:,6]))
      print("dt    ", np.min(x_datas_filtered[:,7]), np.max(x_datas_filtered[:,7]))
      print()
      print("y Xdot", np.min(y_datas_filtered[:,0]), np.max(y_datas_filtered[:,0]))
      print("y Ydot", np.min(y_datas_filtered[:,1]), np.max(y_datas_filtered[:,1]))
      print("y Tdot", np.min(y_datas_filtered[:,2]), np.max(y_datas_filtered[:,2]))
      #print("first 50  ", x_datas_filtered[1000:1050,:])

    #Perform threshold and define threshold array
    if THRESHOLD:

        threshold_array=np.array([0.01, 0.01, np.pi/180, 0, 0,\
                                   0.01, np.pi/180,0])

        x_datas_filtered=threshold_data(x_datas_filtered, threshold_array)
        y_datas_filtered=threshold_data(y_datas_filtered, threshold_array)

    #plt.plot(np.arange(x_datas_filtered.shape[0]-50), x_datas_filtered[50:,index])
    #plt.plot(np.arange(150), x_datas_filtered[50:200,index])
    
    #plt.show() 
    #exit()
    
    if FILTER:
      x_datas = x_datas_filtered
      y_datas = y_datas_filtered
    
    
    ####################################################
    ############       NN Stuff     ####################
    ####################################################
    
    if NN:

    
      dtype = torch.cuda.FloatTensor
      D_in, H, D_out = INPUT_SIZE, 32, OUTPUT_SIZE

      # Shuffle data
      num_samples = x_datas.shape[0]
      rand_idx = np.random.permutation(num_samples)
      x_shuffled = x_datas[rand_idx]
      y_shuffled = y_datas[rand_idx]
      # Make validation set
      split = int(0.9*num_samples)
      x_tr = x_shuffled[:split]
      y_tr = y_shuffled[:split]
      x_val = x_shuffled[split:]
      y_val = y_shuffled[split:]

      #Convert to cuda tensors
      x_tr = torch.from_numpy(x_tr.astype('float32')).type(dtype)
      y_tr = torch.from_numpy(y_tr.astype('float32')).type(dtype)
      x_val = torch.from_numpy(x_val.astype('float32')).type(dtype)
      y_val = torch.from_numpy(y_val.astype('float32')).type(dtype)

      print("validation sets prepared!")

      #Loss function and Learning rate
      loss_fn = torch.nn.MSELoss(size_average=False)
      learning_rate = 0.0001

      #Neural Net
      model=gen_nn()
      model=model.cuda()
      print(model)  # Visualize net architecture

      ##Different Optimizers

      opt_SGD         = torch.optim.SGD(model.parameters(), lr=learning_rate)
      opt_Momentum    = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
      opt_RMSprop     = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
      opt_Adam        = torch.optim.Adam(model.parameters(), lr=learning_rate)
      optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

      #opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

      i = 3 #Specify optimizer
      if LEARN_RESIDUALS:
        filename='/media/JetsonSSD/model_residual.torch'
      else:
        filename='/media/JetsonSSD/model_own.torch'
      do_training(model, filename , optimizers[i], x_tr, y_tr, x_val, y_val, loss_fn, 15000)    
      print('training complete!')
      test_model(model, 10, LEARN_RESIDUALS)


##################################################
