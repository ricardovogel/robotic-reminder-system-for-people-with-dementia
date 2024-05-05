#!/usr/bin/python3
#
# 	@section COPYRIGHT
#	Copyright (C) 2023 Consequential Robotics Ltd
#	
#	@section AUTHOR
#	Consequential Robotics http://consequentialrobotics.com
#	
#	@section LICENSE
#	For a full copy of the license agreement, and a complete
#	definition of "The Software", see LICENSE in the MDK root
#	directory.
#	
#	Subject to the terms of this Agreement, Consequential
#	Robotics grants to you a limited, non-exclusive, non-
#	transferable license, without right to sub-license, to use
#	"The Software" in accordance with this Agreement and any
#	other written agreement with Consequential Robotics.
#	Consequential Robotics does not transfer the title of "The
#	Software" to you; the license granted to you is not a sale.
#	This agreement is a binding legal agreement between
#	Consequential Robotics and the purchasers or users of "The
#	Software".
#	
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
#	KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#	WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
#	OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
#	OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#	

# create node 
import rospy
from std_msgs.msg import UInt8, UInt16, UInt32, Float32MultiArray, UInt16MultiArray, UInt32MultiArray, Int32MultiArray
rospy.init_node("service_pose", anonymous=True)


import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:500"
import sys
import time
import numpy as np
import math

import miro2 as miro
import geometry_msgs

################################################################
#   Camera													   #
################################################################

from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError

#### Person Detection
import torch
from torchvision import transforms

from ultralytics import YOLO

import matplotlib.pyplot as plt
import cv2

################################################################

class controller:
	def callback_cam(self, ros_image, index):
		# silently (ish) handle corrupted JPEG frames
		try:
			# convert compressed ROS image to raw CV image
			image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")

			# store image for display
			self.input_camera[index] = image

		# except CvBridgeError as e:
		except Exception as e:
		# 	# swallow error, silently
		# 	print(e)
			pass

	def callback_caml(self, ros_image):
		self.callback_cam(ros_image, 0)


	def callback_camr(self, ros_image):
		self.callback_cam(ros_image, 1)


	def load_pose_model(self):
		return YOLO('yolov8n-pose.pt')

	def detect_poses(self, images):
		output = self.pose_model(images, conf=0.7, device=self.device) # show=True
		output_l = output[0]
		output_r = output[1]

		nr_of_poses_l = output_l.keypoints.xy.shape[0] if output_l.keypoints.xy.shape[1] != 0 else 0
		nr_of_poses_r = output_r.keypoints.xy.shape[0] if output_r.keypoints.xy.shape[1] != 0 else 0

		return output_l, nr_of_poses_l, output_r, nr_of_poses_r

	def is_standing(self, pose):
		if not self.ignore_sitting_people:
			return True
		upper_leg = (pose[11,1] + pose[12,1])/2
		knee = (pose[14,1] + pose[13,1])/2

		return (abs(upper_leg - knee) > 0.1).item()

	def tensor_to_cv_image(self, image, to_bgr):
		nimg = image[0].permute(1, 2, 0) * 255
		nimg = nimg.cpu().numpy().astype(np.uint8)
		nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
		if not to_bgr:
			nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
		return nimg

	def visualize_pose(self, output, image):
		for pose in output.keypoints.xy:
			for i, point in enumerate(pose):
				x_pos = math.floor(point[0])
				y_pos = math.floor(point[1])
				image = cv2.putText(image, "{}".format(i), (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209, 80, 0, 255), 3)

		return image

	def loop(self):
		show_cameras = True

		msg_pose_det = Int32MultiArray()
		msg_pose_det.data = [0, 0, 0, 0, 0, 0, 0] # [nr of poses L, nr of poses R, center pose 1 L, center pose 1 R, centerness, head height, closeness]

		# loop
		while not rospy.core.is_shutdown():
			try:		
				image_l = self.input_camera[0]
				image_r = self.input_camera[1]

				if not image_l is None and not image_r is None:
						# handle
						self.input_camera[0] = None
						self.input_camera[1] = None

						# print("Pose detection")
						output_l, nr_of_poses_l, output_r, nr_of_poses_r = self.detect_poses([image_l, image_r])

						avg_x_l = 0
						avg_x_r = 0
						head_height = 0
						closeness = 0

						are_poses_standing_l = []
						if output_l.keypoints.xy.shape[1] != 0:
							for pose in output_l.keypoints.xyn:
								are_poses_standing_l.append(self.is_standing(pose))

						nr_of_standing_l = len(list(filter(lambda x: x, are_poses_standing_l)))

						if nr_of_standing_l == 1:
							index_of_stander = are_poses_standing_l.index(True)

							head_height += output_l.keypoints.xyn[index_of_stander][0][1]

							main_body_x = [output_l.keypoints.xy[index_of_stander][5][0], output_l.keypoints.xy[index_of_stander][6][0], output_l.keypoints.xy[index_of_stander][11][0], output_l.keypoints.xy[index_of_stander][12][0]]
							main_body_y = [output_l.keypoints.xy[index_of_stander][5][1], output_l.keypoints.xy[index_of_stander][6][1], output_l.keypoints.xy[index_of_stander][11][1], output_l.keypoints.xy[index_of_stander][12][1]]

							closeness += (max(main_body_x) - min(main_body_x)) + (max(main_body_y) - min(main_body_y))

							avg_x_l = math.floor(sum(main_body_x) / len(main_body_x))

						are_poses_standing_r = []
						if output_r.keypoints.xy.shape[1] != 0:
							for pose in output_r.keypoints.xyn:
								are_poses_standing_r.append(self.is_standing(pose))

						nr_of_standing_r = len(list(filter(lambda x: x, are_poses_standing_r)))
						if nr_of_standing_r == 1:
							index_of_stander = are_poses_standing_r.index(True)

							head_height += output_r.keypoints.xyn[index_of_stander][0][1]

							main_body_x = [output_r.keypoints.xy[index_of_stander][5][0], output_r.keypoints.xy[index_of_stander][6][0], output_r.keypoints.xy[index_of_stander][11][0], output_r.keypoints.xy[index_of_stander][12][0]]
							main_body_y = [output_r.keypoints.xy[index_of_stander][5][1], output_r.keypoints.xy[index_of_stander][6][1], output_r.keypoints.xy[index_of_stander][11][1], output_r.keypoints.xy[index_of_stander][12][1]]

							closeness += (max(main_body_x) - min(main_body_x)) + (max(main_body_y) - min(main_body_y))

							avg_x_r = math.floor(sum(main_body_x) / len(main_body_x))

						if nr_of_standing_l == 1 and nr_of_standing_r == 1:
							head_height /= 2
							closeness /= 2
						
						head_height = math.floor(head_height * 100)
						closeness = math.floor(closeness)
						centerness = (image_r.shape[1] - avg_x_r) - avg_x_l
						
						msg_pose_det.data[0] = nr_of_standing_l
						msg_pose_det.data[1] = nr_of_standing_r
						msg_pose_det.data[2] = avg_x_l
						msg_pose_det.data[3] = avg_x_r
						msg_pose_det.data[4] = centerness
						msg_pose_det.data[5] = head_height
						msg_pose_det.data[6] = closeness

						if show_cameras:
							skeleton_l = self.visualize_pose(output_l, image_l)
							skeleton_r = self.visualize_pose(output_r, image_r)

							if nr_of_standing_l == 1:
								skeleton_l = cv2.circle(skeleton_l, (avg_x_l, 40), 20, (0, 255, 0), -1)
								skeleton_l = cv2.putText(skeleton_l, "cls: {}".format(closeness), (avg_x_l, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209, 80, 0, 255), 3)

							if nr_of_standing_r == 1:
								skeleton_r = cv2.circle(skeleton_r, (avg_x_r, 40), 20, (0, 255, 0), -1)
								skeleton_r = cv2.putText(skeleton_r, "cls: {}".format(closeness), (avg_x_r, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209, 80, 0, 255), 3)

							if nr_of_standing_l == 1 and nr_of_standing_r == 1:
								skeleton_l = cv2.putText(skeleton_l, "ctr: {}".format(centerness), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209, 80, 0, 255), 3)

							
							cv2.imshow("Poses", cv2.vconcat([skeleton_l, skeleton_r]))
							cv2.waitKey(1)

				# update nodes
				self.pub_pose_detection.publish(msg_pose_det)
			
			except Exception as e:
				print(e)
				pass

			# state
			sleep_time = 0.02
			time.sleep(sleep_time)

	def __init__(self, args):

		self.input_camera = [None, None, None]
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("Device:", self.device)
		self.pose_model = self.load_pose_model()

		# state
		self.vbat = 0

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		topic = topic_base_name + "/services/pose_detection"
		print ("publish", topic)
		self.pub_pose_detection = rospy.Publisher(topic, Int32MultiArray, queue_size=0)

		# subscribe
		self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed",
					CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
		self.sub_camr = rospy.Subscriber(topic_base_name + "/sensors/camr/compressed",
					CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

		self.image_converter = CvBridge()


		# wait for connect
		print ("wait for connect...")
		time.sleep(1)

		self.ignore_sitting_people = False
		if "--ignore-sitting-people" in args:
			self.ignore_sitting_people = True


if __name__ == "__main__":
	# normal singular invocation
	main = controller(sys.argv[1:])
	main.loop()

