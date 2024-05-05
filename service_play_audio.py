#!/usr/bin/python3
#
#	@section COPYRIGHT
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

import rospy
from std_msgs.msg import UInt8MultiArray, UInt16MultiArray, Int16MultiArray, String, UInt16
import time
import sys
import os
import numpy as np
import hashlib

# messages larger than this will be dropped by the receiver
MAX_STREAM_MSG_SIZE = (4096 - 48)

# amount to keep the buffer stuffed - larger numbers mean
# less prone to dropout, but higher latency when we stop
# streaming. with a read-out rate of 8k, 2000 samples will
# buffer for quarter of a second, for instance.
BUFFER_STUFF_BYTES = 4000

# media source list
DIR_SOURCE = [
	"./audio/"
	]

# list directories belonging to other releases
DIR_ROOT="../../../"
DIR_MDK = []
subdirs = os.listdir(DIR_ROOT)
for d in subdirs:
	if len(d) < 3:
		continue
	if d[0:3] != "mdk":
		continue
	DIR_MDK.append(d)

# rsort them so we prioritise more recent ones
DIR_MDK.sort()
DIR_MDK.reverse()

# and append to media source list
for d in DIR_MDK:
	DIR_SOURCE.append(DIR_ROOT + d + "/share/media")

# append dev directories
DIR_SOURCE.append(os.getenv('HOME') + "/lib/miro2x/mdk/share/media")

################################################################

def error(msg):

	print(msg)
	sys.exit(0)

################################################################

# index
index = []
index_special = []

# not digits
digits = None

# for each source directory
for dir in DIR_SOURCE:

	# if is directory
	if os.path.isdir(dir):

		# get files
		files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(".mp3")]

		# add to array
		for file in files:

			# special files prepended with underscore
			if file[0] == '_':

				if file[1] != '_':
					index_special.append([file, os.path.join(dir, file)])

			# normal files
			else:

				# append to index
				index.append([file, os.path.join(dir, file)])

		# if any were found, we stop there, because the source
		# directories are intended not to union each other, but
		# to override each other, in the order specified in
		if len(index) > 0:
			print ("reading from:", dir)
			break

# sort array
index.sort()

# if no argument provided
if len(sys.argv) == 1:

	# show index
	print ("pass a numerical index to select a file to stream:")
	for i, item in enumerate(index):
		print ("\t", i+1, "-", item[0])

	# done
	exit()

# if argument is prepended underscore, treat as digit string
if sys.argv[1][0] == '_':

	# digits
	digits = sys.argv[1][1:]

	# load known file
	TRACK_FILE = index_special[0][0]
	TRACK_PATH = index_special[0][1]

# if argument is prepended /, treat as filename
elif sys.argv[1][0] == '/':

	# load specified file
	TRACK_PATH = sys.argv[1]
	TRACK_FILE = hashlib.md5(TRACK_PATH).hexdigest()

# otherwise, choose item from list
else:

	# get index
	i = int(sys.argv[1]) - 1
	if i < 0 or i >= len(index):
		error("item not found in index")

	# and extract its parts
	TRACK_FILE = index[i][0]
	TRACK_PATH = index[i][1]

# if the file is not there, fail
if not os.path.isfile(TRACK_PATH):
	error('file not found');

# report
print ("playing file", TRACK_PATH)



################################################################

class streamer:

	def callback_log(self, msg):

		sys.stdout.write(msg.data)
		sys.stdout.flush()

	def callback_stream(self, msg):

		self.buffer_space = msg.data[0]
		self.buffer_total = msg.data[1]
	
	def callback_play_audio(self, msg):
		print("Callback play audio!", msg)
		if str(msg.data) == sys.argv[1]:
			print("eq")
			main.play_sound(sys.argv[2:])

	def play_sound(self, args):
		# decode mp3
		file = "/tmp/" + TRACK_FILE + ".decode"
		if not os.path.isfile(file):
			cmd = "ffmpeg -y -i \"" + TRACK_PATH + "\" -f s16le -acodec pcm_s16le -ar 8000 -ac 1 \"" + file + "\""
			os.system(cmd)
			if not os.path.isfile(file):
				error('failed decode mp3')

		# load wav
		with open(file, 'rb') as f:
			dat = f.read()
		self.data_r = 0

		# convert to numpy array
		dat = np.frombuffer(dat, dtype='int16')#.astype(np.int32)

		# normalise wav
		dat = dat.astype(np.float64)
		sc = 32767.0 / np.max(np.abs(dat))
		dat *= sc
		dat = dat.astype(np.int16).tolist()

		# store
		self.data = dat

		# state
		self.buffer_space = 0
		self.buffer_total = 0

		state_file = None
		if len(args):
			state_file = args[0]

		# periodic reports
		count = 0

		# safety dropout if receiver not present
		dropout_data_r = -1
		dropout_count = 3

		# loop
		while not rospy.core.is_shutdown():

			# check state_file
			if not state_file is None:
				if not os.path.isfile(state_file):
					break

			# if we've received a report
			if self.buffer_total > 0:

				# compute amount to send
				buffer_rem = self.buffer_total - self.buffer_space
				n_bytes = BUFFER_STUFF_BYTES - buffer_rem
				n_bytes = max(n_bytes, 0)
				n_bytes = min(n_bytes, MAX_STREAM_MSG_SIZE)

				# if amount to send is non-zero
				if n_bytes > 0:

					msg = Int16MultiArray(data = self.data[self.data_r:self.data_r+n_bytes])
					self.pub_stream.publish(msg)
					self.data_r += n_bytes

			# break
			if self.data_r >= len(self.data):
				break

			# report once per second
			if count == 0:
				count = 10
				print ("streaming:", self.data_r, "/", len(self.data), "bytes")

				# check at those moments if we are making progress, also
				if dropout_data_r == self.data_r:
					if dropout_count == 0:
						print ("dropping out because of no progress...")
						break
					print ("dropping out in", str(dropout_count) + "...")
					dropout_count -= 1
				else:
					dropout_data_r = self.data_r

			# count tenths
			count -= 1
			time.sleep(0.1)
		

	def __init__(self):
		# get robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# publish
		topic = topic_base_name + "/control/stream"
		print ("publish", topic)
		self.pub_stream = rospy.Publisher(topic, Int16MultiArray, queue_size=0)

		# subscribe
		topic = topic_base_name + "/platform/log"
		print ("subscribe", topic)
		self.sub_log = rospy.Subscriber(topic, String, self.callback_log, queue_size=5, tcp_nodelay=True)

		# subscribe
		topic = topic_base_name + "/sensors/stream"
		print ("subscribe", topic)
		self.sub_stream = rospy.Subscriber(topic, UInt16MultiArray, self.callback_stream, queue_size=1, tcp_nodelay=True)

		topic = topic_base_name + "/services/play_audio"
		print ("subscribe", topic)
		self.sub_stream = rospy.Subscriber(topic, UInt16, self.callback_play_audio, queue_size=1, tcp_nodelay=True)

if __name__ == "__main__":

	rospy.init_node("client_stream", anonymous=True)
	main = streamer()
	# time.sleep(2.0)
	# main.play_sound(sys.argv[2:])
	while not rospy.core.is_shutdown():
		pass
	
