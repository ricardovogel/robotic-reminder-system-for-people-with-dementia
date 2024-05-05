# Robotic Reminder System for People with Dementia
by Ricardo Vogel

Code for a high-fidelity prototype used in Ricardo Vogel's MSc thesis project. The code runs on the [MiRo-e robot](https://www.miro-e.com/robot). It was used in two design workshops to improve the design presented in the MSc thesis. The prototype robot drives towards the user, gets their attention, and then guides them to a screen. The screen is found using a QR code placed at the robot's eye level. The thesis can be found [here](https://repository.tudelft.nl/islandora/object/uuid%3A2f9b1db8-6cbb-40fb-8947-952ca2032c4c).

## Structure
The code consists of five services:

- **client.py** handles the movement and behavior of the robot. This includes state switching, driving, sounds, and cosmetic elements (e.g., the robot's tail). `--no-cliff-reflex` turns off the (mostly inaccurate) cliff reflex. 
- **service_pose.py** detects people and the positions of their limbs. This is used by the client to center the robot on the person, to look at their face, and to see if they are standing or sitting. `--ignore-sitting-people` makes the system only consider standing people.  
- **service_qr.py** detects QR codes and shares their locations. This is used to find the screen. Note that this needs to be calibrated depending on lighting conditions. Use `--pink-qr`flag if the QR code is printed on bright pink paper. The robot starts spinning slowly when pink is seen, so that motion blur is reduced and the QR code is easier to see.
- **service_play_audio.py** plays an audio file when triggered by the client. It takes audio files from the audio folder. Note that a seperate instance of this program should be ran for each audio file. 
- **client_introduction.py** starts up an introduction. Make sure a service_play_audio instance is running with the introduction file.

# Prerequisites
- ROS
- Python
- OpenCV2
- PyTorch
- NumPy
- MatPlotLib
- Ultralytics (YOLOv8)
- QReader

