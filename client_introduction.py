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
#	Subject to the terms  this Agreement, Consequential
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
rospy.init_node("client", anonymous=True)

import os
import sys
import time
import numpy as np
import math

import miro2 as miro
import geometry_msgs

################################################################
#   Camera													   #
################################################################

from sensor_msgs.msg import CompressedImage, JointState

import cv2
from cv_bridge import CvBridge, CvBridgeError

#### Person Detection
import face_recognition

import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2

################################################################

class controller:

	def loop(self):
		t_now = 0.0
		t_delta = 0.02

		f_cos = 1.0

		msg_cos = Float32MultiArray()
		msg_cos.data = [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]

		while not rospy.core.is_shutdown():
			try:
				xc = math.sin(t_now * f_cos * 2 * math.pi)

				move_tail_towards = xc + 0.5

				if msg_cos.data[1] > move_tail_towards + 0.05:
					msg_cos.data[1] -= 0.1
				elif msg_cos.data[1] < move_tail_towards - 0.05:
					msg_cos.data[1] += 0.1

				self.pub_cos.publish(msg_cos)


			except Exception as e:
				print(e)
				pass
			
			time.sleep(t_delta)
			t_now = t_now + t_delta

	def __init__(self, args):
		# state
		self.vbat = 0

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		topic = topic_base_name + "/control/cosmetic_joints"
		print ("publish", topic)
		self.pub_cos = rospy.Publisher(topic, Float32MultiArray, queue_size=0)

		topic = topic_base_name + "/control/kinematic_joints"
		print ("publish", topic)
		self.pub_kin = rospy.Publisher(topic, JointState, queue_size=0)
		
		topic = topic_base_name + "/services/play_audio"
		print ("publish", topic)
		self.pub_play_audio = rospy.Publisher(topic, UInt16, queue_size=0)

		# wait for connect
		print ("wait for connect...")
		time.sleep(1)


		self.pub_play_audio.publish(1)
		print ("OK")

if __name__ == "__main__":

	# normal singular invocation
	main = controller(sys.argv[1:])
	main.loop()

