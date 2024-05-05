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

################################################################

# to use in MIROcode, copy everything below this line into the
# MIROcode Python editor.
#
# vvvvvv vvvvvv vvvvvv vvvvvv

################################################################

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

import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2

from enum import Enum

################################################################
class RobotState(Enum):
	APPROACH = 1
	GUIDETOLOCATION = 2
	CHECKIFFOLLOWING = 3

class controller:

	def callback_pose_detection(self, ros_pose_det):
		self.pose_detection = ros_pose_det
	
	def callback_qr_detection(self, ros_qr_det):
		self.qr_detection = ros_qr_det


	def callback_package(self, msg):
		# report
		vbat = np.round(np.array(msg.battery.voltage) * 100.0) / 100.0
		if not vbat == self.vbat:
			self.vbat = vbat
			# print ("battery", vbat)

	def is_close_enough(self, closeness, closeness_threshold):
		return closeness > closeness_threshold

	def loop(self):
		t_now = 0.0
		t_delta = 0.02


		msg_wheels = geometry_msgs.msg.TwistStamped()

		msg_animal_state = miro.msg.animal_state()
		# set high ambient sound level to maximize volume
		# (see animal_state.msg for more details)
		msg_animal_state.sound_level = 0.1
		
		wakefulness_max = 1.0

		# enable voice
		msg_animal_state.flags = miro.constants.ANIMAL_EXPRESS_THROUGH_VOICE

		msg_animal_state.emotion.valence = 1.0
		msg_animal_state.emotion.arousal = 1.0

		# Kinematic
		msg_kin = JointState()
		msg_kin.position = [0.0, np.radians(30.0), 0.0, 0.0]

		# Cosmetics
		msg_cos = Float32MultiArray()
		msg_cos.data = [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]

		f_kin = 0.25
		f_cos = 1.0

		spin = 0.0
		spin_max = 0.2
		drive = 0.0
		drive_max = 0.2
		miro_head_height_max = 0.8
		miro_head_height_min = 0.2

		closeness_threshold_person = 175
		centerness_threshold_person = 150

		closeness_threshold_qr = 180
		centerness_threshold_qr = 150

		lost_for_how_long = 0.0
		centered_for = 0.0
		hone_in = 0.0
		honed_in_some_time_ago = 0.0

		close_for_how_long = 0.0
		start_guiding_after = 2.0
		said_something_about_screen = False

		guiding_for = 0.0
		guide_for_time = 8.0

		robot_state = RobotState.APPROACH
		
		# loop
		while not rospy.core.is_shutdown():
			try:
				#### GATHER INFO ####
				xk = math.sin(t_now * f_kin * 2 * math.pi)
				xc = math.sin(t_now * f_cos * 2 * math.pi)
				xcc = math.cos(t_now * f_cos * 2 * math.pi)
				xc2 = math.sin(t_now * f_cos * 1 * math.pi)

				nr_of_poses_l = self.pose_detection.data[0]
				nr_of_poses_r = self.pose_detection.data[1]
				centerness_person = self.pose_detection.data[4]
				person_head_height = self.pose_detection.data[5]
				closeness_person = self.pose_detection.data[6]
				
				no_of_qrs_l = self.qr_detection.data[0]
				no_of_qrs_r = self.qr_detection.data[1]
				centerness_qr = self.qr_detection.data[2]
				closeness_qr = self.qr_detection.data[3]
				pinkness_threshold_reached_qr = self.qr_detection.data[4]


				#### STATE SELECTION ####
				no_of_traits_seen_l = no_of_traits_seen_r = centerness = closeness = centerness_threshold = closeness_threshold = -1

				if robot_state == RobotState.APPROACH or robot_state == RobotState.CHECKIFFOLLOWING:
					# We're trying to find a person
					no_of_traits_seen_l = nr_of_poses_l
					no_of_traits_seen_r = nr_of_poses_r
					centerness = centerness_person
					closeness = closeness_person
					centerness_threshold = centerness_threshold_person
					closeness_threshold = closeness_threshold_person
					spin_max = 0.2

				elif robot_state == RobotState.GUIDETOLOCATION:
					# We're trying to find a QR code
					if pinkness_threshold_reached_qr == 0:
						spin_max = 0.2
					elif pinkness_threshold_reached_qr == 1:
						spin_max = 0.13
					else:
						spin_max = 0.1

					no_of_traits_seen_l = no_of_qrs_l
					no_of_traits_seen_r = no_of_qrs_r
					centerness = centerness_qr
					closeness = closeness_qr
					centerness_threshold = centerness_threshold_qr
					closeness_threshold = closeness_threshold_qr			


				##### STATE SWITCHING #####
								
				# APPROACH -> GUIDETOLOCATION
				# Person is close enough and has been close enough for a while
				if robot_state == RobotState.APPROACH and close_for_how_long > start_guiding_after and self.is_close_enough(closeness, closeness_threshold):
					print("STATE: GUIDETOLOCATION")
					robot_state = RobotState.GUIDETOLOCATION
					guiding_for = 0.0
					close_for_how_long = 0.0 # Reset for next time
					self.pub_play_audio.publish(2) # Say "follow me"
					continue
				
				# GUIDETOLOCATION -> CHECKIFFOLLOWING
				# If guiding for enough time, check if following instead
				if robot_state == RobotState.GUIDETOLOCATION and guiding_for >= guide_for_time:
					print("STATE: CHECKIFFOLLOWING")
					robot_state = RobotState.CHECKIFFOLLOWING
					guiding_for = 0.0
					continue
				
				# CHECKIFFOLLOWING -> GUIDETOLOCATION
				# Follow confirmation: person seen in both eyes & enough in the center. We move back to guide to location and we make the robot spin the other way.
				if robot_state == RobotState.CHECKIFFOLLOWING and no_of_traits_seen_l > 0 and no_of_traits_seen_r > 0 and centerness > centerness_threshold * 0.8:
					spin = -1 * spin
					print("STATE: GUIDETOLOCATION")
					robot_state = RobotState.GUIDETOLOCATION
					guiding_for = 0.0
					self.pub_play_audio.publish(2) # Say "follow me"
					continue


				#### LOST ####
				if no_of_traits_seen_l > 0 or no_of_traits_seen_r > 0:
					# If the object used to be lost, it no longer is
					lost_for_how_long = 0.0
				if no_of_traits_seen_l == 0 and no_of_traits_seen_r == 0:
					lost_for_how_long += t_delta
				

				#### CLOSE ENOUGH COUNTER ####
				if no_of_traits_seen_l == 1 and no_of_traits_seen_r == 1 and centerness < centerness_threshold and centerness > -1 * centerness_threshold and self.is_close_enough(closeness, closeness_threshold):
					close_for_how_long += t_delta
				else:
					close_for_how_long -= t_delta

				if close_for_how_long < 0.0:
					close_for_how_long = 0.0 	
				
				if close_for_how_long > start_guiding_after and robot_state == RobotState.GUIDETOLOCATION and said_something_about_screen is False:
					said_something_about_screen = True
					self.pub_play_audio.publish(3) # Say "look at the screen"
					

				#### HONING ####
				if hone_in <= 0.1 and hone_in >= 0.05:
					honed_in_some_time_ago = 1.0
					hone_in = 0.0

				if hone_in < 0.0:
					hone_in = 0.0
				
				if centered_for < 0.0:
					centered_for = 0.0
				
				honed_in_some_time_ago -= t_delta
					
				if no_of_traits_seen_l == 1 and no_of_traits_seen_r == 1 and centerness < centerness_threshold and centerness > -1 * centerness_threshold:
					if centered_for > 1:
						centered_for = 0.0
						hone_in = 2.0
					else:
						centered_for += t_delta
				else:
					centered_for -= t_delta
					hone_in -= t_delta
				
				currently_honing = hone_in > 0.005
				

				##### SPIN #####
				if no_of_traits_seen_l > 0 and no_of_traits_seen_r > 0: # Seen in both eyes
					if  no_of_traits_seen_l == 1 and no_of_traits_seen_r == 1:
						if centerness > centerness_threshold:
							# Target not in center, spin
							spin += 0.01
						elif centerness < -1 * centerness_threshold:
							spin -= 0.01
						else:
							# target is centered, start honing timer after a while, keep honing if already
							if spin > 0.05:
								spin -= 0.05
							elif spin < -0.05:
								spin += 0.05
							else:
								spin = 0.0					
					else:
						# If more than one target seen: slow down
						if spin > 0.05:
							spin -= 0.05
						elif spin < -0.05:
							spin += 0.05
						else:
							spin = 0.0
				elif no_of_traits_seen_l > 0 and no_of_traits_seen_r == 0: # Seen in left eye only 
					spin += 0.05
				elif no_of_traits_seen_l == 0 and no_of_traits_seen_r > 0: # Seen in right eye only
					spin -= 0.05
				elif no_of_traits_seen_l == 0 and no_of_traits_seen_r == 0: # Lost!
					if lost_for_how_long > 1.0:
						# Only start speeding up after a while
						if spin > 0.0:
							spin += 0.05
						else:
							spin -= 0.05

				if spin > spin_max:
					spin = spin_max
				if spin < -1.0 * spin_max:
					spin = -1.0 * spin_max
				if currently_honing:
					# Stop spinning if honing
					spin = 0.0
				msg_wheels.twist.angular.z = spin * 6.2832


				#### DRIVING ####
				if  no_of_traits_seen_l == 1 and no_of_traits_seen_r == 1 and centerness < centerness_threshold and centerness > -1 * centerness_threshold and not self.is_close_enough(closeness, closeness_threshold):
					drive += 0.05
				else:
					drive -= 0.05

				if abs(spin) > 0.05:
					# Stop driving if spinning
					drive = 0.0
				if drive > drive_max:
					drive = drive_max
				if drive < 0.0:
					drive = 0.0

				msg_wheels.twist.linear.x = drive
				

				##### SOUND ##### (MiRo Built-in Sounds)
				# Default: no sound, high valence, high arousal
				msg_animal_state.sleep.wakefulness = 0.0
				msg_animal_state.emotion.valence = 1.0
				msg_animal_state.emotion.arousal = 1.0
				
				if self.is_close_enough(closeness, closeness_threshold):
					# Start making sound if close to target (whether person or QR)
					msg_animal_state.sleep.wakefulness = wakefulness_max

				if no_of_traits_seen_l == 0 and no_of_traits_seen_r == 0 and lost_for_how_long > 1.0 and robot_state == RobotState.APPROACH or robot_state == RobotState.CHECKIFFOLLOWING: 
					# Become sad if target not seen, and hasn't been seen for a while
					msg_animal_state.sleep.wakefulness = wakefulness_max
					msg_animal_state.emotion.valence = 0.0
					msg_animal_state.emotion.arousal = 0.5
				
				if honed_in_some_time_ago > 0.005 and robot_state == RobotState.APPROACH:
					# Recently hones in but not anymore, so currently turning slightly
					msg_animal_state.sleep.wakefulness = wakefulness_max


				##### KINETICS (head movement) #####
				# Head height is smoothed, instead of quickly moving to the correct position, the position is altered until it is correct
				move_head_height_towards = np.radians(30.0)
				if (robot_state == RobotState.APPROACH or robot_state == RobotState.CHECKIFFOLLOWING) and (no_of_traits_seen_l > 0 or no_of_traits_seen_r > 0):
					move_head_height_towards = person_head_height / 100
				
				# Adjust head height
				if msg_kin.position[1] > move_head_height_towards + 0.05:
					msg_kin.position[1] -= 0.01
				elif msg_kin.position[1] < move_head_height_towards - 0.05:
					msg_kin.position[1] += 0.01
					
				if msg_kin.position[1] > miro_head_height_max:
					msg_kin.position[1] = miro_head_height_max
				if msg_kin.position[1] < miro_head_height_min:
					msg_kin.position[1] = miro_head_height_min
			
				
				##### COSMETICS (e.g., tail, ears) #####
				move_tail_towards = 0.5
				if no_of_traits_seen_l > 0 or no_of_traits_seen_r > 0 or robot_state == RobotState.GUIDETOLOCATION:
					# Wag tail if target seen, or if guiding to location (excited to be guiding)
					move_tail_towards = xc + 0.5

				if msg_cos.data[1] > move_tail_towards + 0.05:
					msg_cos.data[1] -= 0.1
				elif msg_cos.data[1] < move_tail_towards - 0.05:
					msg_cos.data[1] += 0.1

				# update nodes
				self.pub_animal_state.publish(msg_animal_state)
				self.pub_cos.publish(msg_cos)
				self.pub_wheels.publish(msg_wheels)
				self.pub_kin.publish(msg_kin)



			except Exception as e:
				print(e)
				pass

			# state
			time.sleep(t_delta)
			t_now = t_now + t_delta

	def __init__(self, args):
		self.face_detection = UInt32MultiArray()
		self.face_detection.data = [0, 0, 0, 0]
		self.pose_detection = Int32MultiArray()
		self.pose_detection.data = [0, 0, 0, 0, 0, 0, 0] # [nr of poses L, nr of poses R, center pose 1 L, center pose 1 R, centerness, head height, closeness]
												   # Center of poses, centerness, and head height are only defined if nr of poses L = R = 1.
												   # Centerness is 0 if person in middle, positive if more towards the robot's left, negative if more towards teh robot's right.
												   # Head height is 0 if head high (above screen), 100 if head is low (below screen).
												   # Closeness is closer to 0 if further away, 100 if closer.
		self.qr_detection = Int32MultiArray()
		self.qr_detection.data = [0, 0, 0, 0, 0]

		# state
		self.vbat = 0

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# publish
		topic = topic_base_name + "/control/cmd_vel"
		print ("publish", topic)
		self.pub_wheels = rospy.Publisher(topic, geometry_msgs.msg.TwistStamped, queue_size=0)

		topic = topic_base_name + "/core/animal/state"
		print ("publish", topic)
		self.pub_animal_state = rospy.Publisher(topic, miro.msg.animal_state,
					queue_size=0)

		topic = topic_base_name + "/control/cosmetic_joints"
		print ("publish", topic)
		self.pub_cos = rospy.Publisher(topic, Float32MultiArray, queue_size=0)

		topic = topic_base_name + "/control/kinematic_joints"
		print ("publish", topic)
		self.pub_kin = rospy.Publisher(topic, JointState, queue_size=0)

		topic = topic_base_name + "/control/flags"
		print ("publish", topic)
		self.pub_flags = rospy.Publisher(topic, UInt32, queue_size=0)

		topic = topic_base_name + "/services/play_audio"
		print ("publish", topic)
		self.pub_play_audio = rospy.Publisher(topic, UInt16, queue_size=0)


		# subscribe
		topic = topic_base_name + "/sensors/package"
		print ("subscribe", topic)
		self.sub_package = rospy.Subscriber(topic, miro.msg.sensors_package,
					self.callback_package, queue_size=1, tcp_nodelay=True)

		self.sub_pose_detection = rospy.Subscriber(topic_base_name + "/services/pose_detection", 
					Int32MultiArray, self.callback_pose_detection, queue_size=1, tcp_nodelay=True)
		self.sub_qr_detection = rospy.Subscriber(topic_base_name + "/services/qr_detection", 
					Int32MultiArray, self.callback_qr_detection, queue_size=1, tcp_nodelay=True)		

		# wait for connect
		print ("wait for connect...")
		time.sleep(1)

		msg = UInt32()
		msg.data = miro.constants.PLATFORM_D_FLAG_DISABLE_STATUS_LEDS
		if "--no-cliff-reflex" in args:
			msg.data |= miro.constants.PLATFORM_D_FLAG_DISABLE_CLIFF_REFLEX # Turn off cliff reflex on some floors
		
		print ("send control flags... ", hex(msg.data))
		self.pub_flags.publish(msg)
		print ("OK")

if __name__ == "__main__":

	# normal singular invocation
	main = controller(sys.argv[1:])
	main.loop()

