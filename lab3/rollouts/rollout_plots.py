#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

nothing=np.loadtxt('txt_files/nothing.txt')
forward=np.loadtxt('txt_files/forward.txt')
backward=np.loadtxt('txt_files/backward.txt')
forward_leftsteer=np.loadtxt('txt_files/forward_leftsteer.txt')
forward_rightsteer=np.loadtxt('txt_files/forward_rightsteer.txt')
reverse_leftsteer=np.loadtxt('txt_files/reverse_leftsteer.txt')
reverse_rightsteer=np.loadtxt('txt_files/reverse_rightsteer.txt')

# nothing=np.loadtxt('txt_files/nothing_rl.txt')
# forward=np.loadtxt('txt_files/forward_rl.txt')
# backward=np.loadtxt('txt_files/backward_rl.txt')
# forward_leftsteer=np.loadtxt('txt_files/forward_leftsteer_rl.txt')
# forward_rightsteer=np.loadtxt('txt_files/forward_rightsteer_rl.txt')

def add_with_arrows(data, color, label, plot_arrows=True):
	arrow_length = 20
	# angles = data[:,2].cumsum()
	angles = -data[:,2]
	arrow_xs = np.cos(angles + np.pi/2.0)*arrow_length
	arrow_ys = np.sin(angles + np.pi/2.0)*arrow_length
	plt.plot(data[:,1], data[:,0], c=color, label=label, linewidth=1)
	if plot_arrows:
		plt.quiver(data[:,1], data[:,0], arrow_xs, arrow_ys, color=color, units='inches', scale_units='inches', angles='xy', width=0.02)


add_with_arrows(nothing, 'black', 'Nothing', False)
add_with_arrows(forward, 'g', 'Forward')
add_with_arrows(backward, 'm', 'Backward')
add_with_arrows(forward_leftsteer, 'b' , 'Left steer')
add_with_arrows(forward_rightsteer, 'r', 'Right steer')
add_with_arrows(reverse_leftsteer, 'b' , None)
add_with_arrows(reverse_rightsteer, 'r', None)

plt.title('Neural network rollouts for 10 time steps')
plt.xlabel('Y axis in global frame')
plt.ylabel('X axis in global frame')

legend = plt.legend(loc='upper left', frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_alpha(0.7)
plt.show()

