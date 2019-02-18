#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
from geometry_msgs.msg import PointStamped, Point


class FisheyeLineFollower:
	def __init__(self, pid_pub_topic, sub_topic, pub_topic):
		
		self.bridge=CvBridge()
		self.fisheyeimage_pub=rospy.Publisher(pub_topic, Image, queue_size=1)
		self.fisheyeimage_sub=rospy.Subscriber(sub_topic, Image, self.callback)
		self.pub = rospy.Publisher(pid_pub_topic, PointStamped, queue_size=1)
		
	def callback(self, cam_data):
		try:
			cv_image=self.bridge.imgmsg_to_cv2(cam_data, "passthrough")
			#rospy.logerr("Happy")
		except CvBridgeError as e:
			print e
			rospy.logerr("Sad")
		#All processing here
		
		cropped_image = cv_image
		blurred_image=self.apply_smoothing(cropped_image)
		edge_detected_image=self.edge_detection(blurred_image)
		#semi_processed_image = self.set_region(edge_detected_image, 0.4, 0.55, 0.1, 0.6, 0.55, 0.6, 0.9, 0.6)
		semi_processed_image=self.set_region(edge_detected_image, 0.3, 0.5, 0.05, 0.7, 0.6, 0.55, 0.95, 0.7)
		center_x, center_y, new_image = self.find_center(semi_processed_image)

		dummy_cropped_image=np.zeros((480,640,3), dtype=np.int8)
		dummy_cropped_image[:,:,0]=cropped_image
		dummy_cropped_image[:,:,1]=cropped_image
		dummy_cropped_image[:,:,2]=cropped_image

		"""
		dummy_cropped_image[nonzero_coords, 0]=0
		dummy_cropped_image[nonzero_coords,1]=0
		dummy_cropped_image[nonzero_coords,2]=255
		"""
		
		#blurred_image=self.apply_smoothing(cv_image)
		#roi_image=self.set_region(edge_detected_image)


		#print np.amin(final_processed_image)
		#print np.amax(final_processed_image)

		#print final_processed_image.shape
		#print np.count_nonzero(final_processed_image)
		    
		final_processed_image=new_image
		#final_processed_image=cv2.addWeighted(dummy_cropped_image, 0.4, new_image, 0.9, 0.0)
		#final_processed_image=dummy_cropped_image
		
		pubready_image=self.bridge.cv2_to_imgmsg(final_processed_image, "passthrough")
		
		msg=PointStamped()
		msg.point=Point(center_x,center_y,0)
		msg.header.stamp = rospy.get_rostime()
		
		try:
			self.fisheyeimage_pub.publish(pubready_image)
			self.pub.publish(msg)
		except CvBridgeError as e:	
			print e
		

	def filter_region(self, image, vertices):

		mask=np.zeros_like(image)
		#rospy.logerr("I am inside filter_region")
		if len(mask.shape)==2:
			cv2.fillPoly(mask, vertices, 255)
		else:
			cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
		return cv2.bitwise_and(image, mask)

	def set_region(self, image, blc, blr, tlc, tlr, brc, brr, trc, trr):

		rows, cols=image.shape[:2]
		#rospy.logerr("I am inside set region")
		bottom_left = np.rint([cols*blc, rows*blr])  #bottom right
		top_left=np.rint([cols*tlc, rows*tlr])  #top right
		bottom_right=np.rint([cols*brc, rows*brr])  #bottom left
		top_right=np.rint([cols*trc, rows*trr])  #top left 

		vertices=np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
		return self.filter_region(image, vertices)
		#return vertices


	def hough_lines(self,image):
		"""
		Image to be output of a Canny transform
		"""

		return cv2.HoughLinesP(image=image, rho=1, theta=np.pi/180, threshold=80, minLineLength=20, maxLineGap=30)


	def apply_smoothing(self, image, kernel_size=29):
		"""
		kernel size to be positive and odd
		"""	
		#rospy.logerr("I am applying smoothing")

		return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

	def draw_lines(self, image, lines, color=[255,0,0], thickness=15):
		if lines is None:
			return 
		image=np.copy(image)

		new_image=np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int8)
		
		new_image[:,:,2]=image[:,:]
		
		line_image=np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int8)

		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)


		new_image=cv2.addWeighted(new_image, 0.8, line_image, 1.0, 0.0)
		return np.asarray(new_image)


	def edge_detection(self, image, low_threshold=50, high_threshold=150):
		edges=cv2.Canny(image, low_threshold, high_threshold)

		"""

		plt.subplot(121), plt.imshow(data, cmap='gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122), plt.imshow(edges, cmap='gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		
		plt.show()
		
		"""

		#rospy.logerr("I am detecting edges yo")
		return np.asarray(edges, dtype="uint8") 

		
	def find_center(self, image):

		nonzero_coords=np.nonzero(image)
		nonzero_coords=np.asarray(nonzero_coords)


		
		#print nonzero_coords.shape
		center_x=np.sum(nonzero_coords[0,:])/np.size(nonzero_coords[0,:])
		center_y=np.sum(nonzero_coords[1,:])/np.size(nonzero_coords[1,:])

		new_image=np.zeros((480,640,3), dtype=np.int8)
		new_image[:,:,0]=0
		new_image[:,:,1]=0
		new_image[:,:,2]=image

		new_image[:,318:323,1] = 255
		new_image[:,318:323,0] = 0
		new_image[:,318:323,2] = 0

		new_image[center_x-5:center_x+6,center_y-5:center_y+6,2]=255
		new_image[center_x-5:center_x+6,center_y-5:center_y+6,0]=0
		new_image[center_x-5:center_x+6,center_y-5:center_y+6,1]=0


		#rospy.logerr("My work is almost done here in find center")

		return center_x, center_y, new_image

		
		

if __name__ == "__main__":

	rospy.init_node('my_lf', anonymous=True)

	sub_topic=rospy.get_param('sub_topic')
	pub_topic=rospy.get_param('pub_topic')
	pid_pub_topic=rospy.get_param('pid_pub_topic')
	
	
	my_lf=FisheyeLineFollower(pid_pub_topic,sub_topic, pub_topic)
	

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Shutting down"
	cv2.destroyAllWindows()	


