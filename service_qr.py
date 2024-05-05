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
rospy.init_node("service_qr", anonymous=True)

################################################################

# to use in MIROcode, copy everything below this line into the
# MIROcode Python editor.
#
# vvvvvv vvvvvv vvvvvv vvvvvv

###################################################uint32#############

import os
import sys
import time
import numpy as np
import math

import miro2 as miro
import geometry_msgs

from qreader import QReader

################################################################
#   Camera													   #
################################################################

from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError

#### Person Detection
import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

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

	def calculate_pinkness(self, image):
		img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		lower = np.array([347/2 - 15, 100, 100])
		upper = np.array([347/2 + 15,255,255])
		mask = cv2.inRange(img, lower, upper)
		total_pixels = img.shape[0] * img.shape[1]
		pink_pixel_count = cv2.countNonZero(mask)
		pinkness = int((pink_pixel_count / total_pixels)*1000)/1000

		return pinkness

	def loop(self):
		show_cameras = True

		msg_qr_det = Int32MultiArray()
		msg_qr_det.data = [0, 0, 0, 0, 0] # [no_l, no_r, centerness, closeness, pinkness_threshold_reached (1 or 0 if pink_qr is True, -1 if pink_qr is False)]
    
        qreader = QReader(model_size='l', min_confidence=0.3) # NOTE: change depending on lighting conditions.

		# loop
		while not rospy.core.is_shutdown():
			try:
				image_l = self.input_camera[0]
				image_r = self.input_camera[1]

				if not image_l is None and not image_r is None:
						# handle
						self.input_camera[0] = None
						self.input_camera[1] = None

						if self.pink_qr:
							pinkness_l = self.calculate_pinkness(image_l)
							pinkness_r = self.calculate_pinkness(image_r)
                            pinkness_threshold = 0.004 # NOTE: change depending on lighting conditions.
							if pinkness_l > pinkness_threshold or pinkness_r > pinkness_threshold:
								msg_qr_det.data[4] = 1
							else:
								msg_qr_det.data[4] = 0
							
						else:
							msg_qr_det.data[4] = -1

						qr_codes_l = qreader.detect(image=image_l, is_bgr=False)
						qr_codes_r = qreader.detect(image=image_r, is_bgr=False)

						nr_of_qrs_l = len(qr_codes_l)
						nr_of_qrs_r = len(qr_codes_r)

						centerness = 0
						closeness = 0
						if nr_of_qrs_l == 1:
							closeness += math.floor(sum(qr_codes_l[0]["wh"]))
						if nr_of_qrs_r == 1:
							closeness += math.floor(sum(qr_codes_r[0]["wh"]))

						if nr_of_qrs_l == 1 and nr_of_qrs_r == 1:
							closeness = math.floor(closeness/2)
							centerness = math.floor((image_l.shape[1] - qr_codes_r[0]["cxcy"][0]) - qr_codes_l[0]["cxcy"][0])

						msg_qr_det.data[0] = nr_of_qrs_l
						msg_qr_det.data[1] = nr_of_qrs_r
						msg_qr_det.data[2] = centerness
						msg_qr_det.data[3] = closeness

						if show_cameras:
							image_l = cv2.putText(image_l, "cls: {}".format(closeness), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 255), 3)
							if nr_of_qrs_l == 1:
								image_l = cv2.rectangle(image_l, (math.floor(qr_codes_l[0]["bbox_xyxy"][0]), math.floor(qr_codes_l[0]["bbox_xyxy"][1])), (math.floor(qr_codes_l[0]["bbox_xyxy"][2]), math.floor(qr_codes_l[0]["bbox_xyxy"][3])), color=(255,0,0), thickness=3)
								image_l = cv2.putText(image_l, "ctr: {}".format(centerness), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 255), 3)

							if nr_of_qrs_r == 1:
								image_r = cv2.rectangle(image_r, (math.floor(qr_codes_r[0]["bbox_xyxy"][0]), math.floor(qr_codes_r[0]["bbox_xyxy"][1])), (math.floor(qr_codes_r[0]["bbox_xyxy"][2]), math.floor(qr_codes_r[0]["bbox_xyxy"][3])), color=(255,0,0), thickness=3)

							cv2.imshow("QR Codes", cv2.vconcat([image_l, image_r]))
							cv2.waitKey(1)
					
				# update nodes
				self.pub_qr_detection.publish(msg_qr_det)
			
			except Exception as e:
				print(e)
				pass

			# state
			sleep_time = 0.02
			time.sleep(sleep_time)

	def __init__(self, args):

		self.input_camera = [None, None, None]

		# state
		self.vbat = 0

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		topic = topic_base_name + "/services/qr_detection"
		print ("publish", topic)
		self.pub_qr_detection = rospy.Publisher(topic, Int32MultiArray, queue_size=0)

		# subscribe
		self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed",
					CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
		self.sub_camr = rospy.Subscriber(topic_base_name + "/sensors/camr/compressed",
					CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

		self.image_converter = CvBridge()

		# wait for connect
		print ("wait for connect...")
		time.sleep(1)

		self.pink_qr = False
		if "--pink-qr" in args:
			self.pink_qr = True


if __name__ == "__main__":
	# normal singular invocation
	main = controller(sys.argv[1:])
	main.loop()

