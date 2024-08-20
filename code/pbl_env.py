## License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
## Applied to sections that are not already under other licenses as specified.

import os
import sys
import numpy as np
import cv2
import pybullet as p
import pybullet_data
from contextlib import contextmanager
from time import sleep
from PIL import Image
np.set_printoptions(precision=4, edgeitems=10, linewidth=100)
from IPython.display import display, clear_output


class PBL(object):
	output_shape = None
	name = 'pbl'
	
	"""docstring for PBL"""
	def __init__(self, input_shape = (120, 120, 3), joints=None, skip_frame=None):
		self.input_shape = input_shape
		self.skip_frame=120
		#cam stuff
		self.fov = 60
		self.aspect = 1
		self.near = 0.2
		self.far = 100
		self.ham_urdf = "pbl_resources/ham.urdf"
		self.pr2_urdf = "pbl_resources/pr2/pr2.urdf"
		self.cheese_urdf = "pbl_resources/cheese.obj"
		self.toast_urdf = "pbl_resources/toast.urdf"
		self.maxForce = 500
		self.all_joints = {
			'''
			'l_shoulder_pan_joint': 59, 
			'l_shoulder_lift_joint': 60, 
			'l_upper_arm_roll_joint': 61, 
			'l_upper_arm_joint': 62, 
			'l_elbow_flex_joint': 63, 
			'l_forearm_roll_joint': 64, 
			'l_forearm_joint': 65, 
			'l_wrist_flex_joint': 66, 
			'l_wrist_roll_joint': 67, 
			'l_gripper_palm_joint': 68, 
			'l_gripper_led_joint': 69, 
			'l_gripper_tool_joint': 71, 
			'l_gripper_l_finger_joint': 72, 
			'l_gripper_l_finger_tip_joint': 73, 
			'l_gripper_r_finger_joint': 74, 
			'l_gripper_r_finger_tip_joint': 75, 
			'l_gripper_joint': 76, 
			'''
			'r_shoulder_pan_joint': 37, 
			'r_shoulder_lift_joint': 38, 
			'r_upper_arm_roll_joint': 39, 
			'r_upper_arm_joint': 40, 
			'r_elbow_flex_joint': 41, 
			'r_forearm_roll_joint': 42, 
			'r_forearm_joint': 43, 
			'r_wrist_flex_joint': 44, 
			'r_wrist_roll_joint': 45, 
			'r_gripper_palm_joint': 46, 
			'r_gripper_led_joint': 47, 
			'r_gripper_tool_joint': 49, 
			'r_gripper_l_finger_joint': 50, 
			'r_gripper_l_finger_tip_joint': 51, 
			'r_gripper_r_finger_joint': 52, 
			'r_gripper_r_finger_tip_joint': 53, 
			'r_gripper_joint': 54, 
		}
		self.joints = list(self.all_joints.values())
		self.output_shape = len(self.joints)
		#['r_shoulder_pan_joint': 37, 'r_shoulder_lift_joint': 38]
		return

	def setup(self):
		if p.isConnected():
			p.disconnect()
			sleep(0.01)
		self.physicsClient = p.connect(p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.8)
		with self.suppress_stdout():
			self.planeId = p.loadURDF("plane.urdf", [0,0,0])
			self.box = p.loadURDF("cube.urdf", basePosition=[0,0,0.85], globalScaling=1.7, useMaximalCoordinates = True)
			self.toast = p.loadURDF(self.toast_urdf, basePosition=[0.4,0.2,1.73], globalScaling=0.35)
			self.cheese = p.loadSoftBody(self.cheese_urdf, basePosition=[0.3,0.2,1.75], baseOrientation = [1,0,0,1], scale=0.2)
			self.ham = p.loadURDF(self.ham_urdf, basePosition=[0.6,-0.6,2.7], globalScaling=0.2, flags= p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT|p.URDF_USE_IMPLICIT_CYLINDER, baseOrientation = [1,0,0,1])
			self.pr2 = p.loadURDF(self.pr2_urdf, basePosition=[-2.0,0,0], globalScaling=3, flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

		self.camPos, self.cam_orn, _, _, _, _ = p.getLinkState(self.pr2, 21) # 21 is the number I got from print(_joint_or_link_name_to_id(pr2,'link'))
		self.view_matrix = p.computeViewMatrix(np.asarray(self.camPos)+[1,0,0], [0.5, 0, 1], [1, 0, 0])
		self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

		p.setRealTimeSimulation(1)
		p.changeDynamics(self.box,-1, mass=100)
		p.changeVisualShape(self.toast, -1, rgbaColor=[1,0.9,0.8,1])
		p.changeVisualShape(self.cheese, -1, rgbaColor=[1,0.7,0,1])
		p.changeVisualShape(self.ham, -1, rgbaColor=[1,0.5,0.6,1])
		### Moving left the arm up #l_shoulder_lift_joint
		p.setJointMotorControl2(bodyUniqueId=self.pr2, jointIndex=60, controlMode=p.VELOCITY_CONTROL, targetVelocity=-1, force=self.maxForce)

		for i in range(self.skip_frame):
			p.stepSimulation()
			scene = self.state()
			#cv2.namedWindow("state", cv2.WINDOW_NORMAL)
			#cv2.imshow('state', cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
			#cv2.waitKey(1)

		hamPos, _ = p.getBasePositionAndOrientation(self.ham)
		cheesePos, _ = p.getBasePositionAndOrientation(self.cheese)
		self.original_distance = np.linalg.norm(np.asarray(cheesePos) - np.asarray(hamPos))
		self.policy = np.random.uniform(-1.0,1.0,self.output_shape)
		self.play(self.policy)
		return

	def test_setup(self):
		images = []
		images2 = []
		if p.isConnected():
			p.disconnect()
			sleep(0.01)
		self.physicsClient = p.connect(p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.8)
		#with self.suppress_stdout():
		self.planeId = p.loadURDF("plane.urdf", [0,0,0])
		self.box = p.loadURDF("cube.urdf", basePosition=[0,0,0.85], globalScaling=1.7, useMaximalCoordinates = True)
		self.toast = p.loadURDF(self.toast_urdf, basePosition=[0.4,0.2,1.73], globalScaling=0.35)
		self.cheese = p.loadSoftBody(self.cheese_urdf, basePosition=[0.3,0.2,1.75], baseOrientation = [1,0,0,1], scale=0.2)
		self.ham = p.loadURDF(self.ham_urdf, basePosition=[0.6,-0.6,2.7], globalScaling=0.2, flags= p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT|p.URDF_USE_IMPLICIT_CYLINDER, baseOrientation = [1,0,0,1])
		self.pr2 = p.loadURDF(self.pr2_urdf, basePosition=[-2.0,0,0], globalScaling=3, flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

		self.camPos, self.cam_orn, _, _, _, _ = p.getLinkState(self.pr2, 21) # 21 is the number I got from print(_joint_or_link_name_to_id(pr2,'link'))
		self.view_matrix = p.computeViewMatrix(np.asarray(self.camPos)+[1,0,0], [0.5, 0, 1], [1, 0, 0])
		self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

		p.setRealTimeSimulation(1)
		p.changeDynamics(self.box,-1, mass=100)
		p.changeVisualShape(self.toast, -1, rgbaColor=[1,0.9,0.8,1])
		p.changeVisualShape(self.cheese, -1, rgbaColor=[1,0.7,0,1])
		p.changeVisualShape(self.ham, -1, rgbaColor=[1,0.5,0.6,1])
		### Moving left the arm up #l_shoulder_lift_joint
		p.setJointMotorControl2(bodyUniqueId=self.pr2, jointIndex=60, controlMode=p.VELOCITY_CONTROL, targetVelocity=-1, force=self.maxForce)

		for i in range(self.skip_frame):
			p.stepSimulation()
			scene = self.state()
			#cv2.imshow('state', cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
			#cv2.waitKey(1)
			display(Image.fromarray(scene).resize((300, 300)))
			clear_output(wait=True)

			images.append(Image.fromarray(scene))
			images2.append(Image.fromarray(self.observation()))
		
		hamPos, _ = p.getBasePositionAndOrientation(self.ham)
		cheesePos, _ = p.getBasePositionAndOrientation(self.cheese)
		self.original_distance = np.linalg.norm(np.asarray(cheesePos) - np.asarray(hamPos))
		
		self.policy = np.random.uniform(-1.0,1.0,self.output_shape)
		self.play(self.policy)
		images.append(Image.fromarray(self.state()))
		images2.append(Image.fromarray(self.observation()))
		return self.original_distance, images, images2

	def state(self):
		img = p.getCameraImage(self.input_shape[0],self.input_shape[1], 
			viewMatrix=self.view_matrix, 
			projectionMatrix=self.projection_matrix, 
			renderer=p.ER_TINY_RENDERER, 
			flags=p.ER_NO_SEGMENTATION_MASK)
		img = np.array(img[2])[:,:,0:3] # cut away alpha channel
		return img

	def observation(self):
		self.view_matrix2 = p.computeViewMatrix([3,0,3], [-2.5, 0, 1.5], [1, 0, 1])
		img = p.getCameraImage(600, 400,
			viewMatrix=self.view_matrix2, 
			projectionMatrix=self.projection_matrix, 
			renderer=p.ER_TINY_RENDERER, 
			flags=p.ER_NO_SEGMENTATION_MASK)
		img = np.array(img[2])
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img
			
	def play(self, exp_action):
		forces = np.full((self.output_shape), self.maxForce)
		p.setJointMotorControlArray(bodyUniqueId=self.pr2, jointIndices=self.joints, 
			controlMode=p.VELOCITY_CONTROL, targetVelocities = exp_action, forces=forces)
		for _ in range(3):
			p.stepSimulation()
			sleep(0.01)
		scene = self.state()
		#cv2.imshow('state', cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
		#cv2.waitKey(1)
		display(Image.fromarray(scene).resize((300, 300)))
		clear_output(wait=True)
		return 

	def reward(self, step):
		step += 1
		hamPos, _ = p.getBasePositionAndOrientation(self.ham)
		cheesePos, _ = p.getBasePositionAndOrientation(self.cheese)
		dist = np.linalg.norm(np.asarray(cheesePos) - np.asarray(hamPos))
		travelled = (self.original_distance - dist) * 100
		reward = travelled / step  # So the timestep is reducing the reward
		return reward

	def close(self):
		cv2.destroyAllWindows()
		if p.isConnected():
			p.disconnect()
			sleep(0.01)
		return True

	@contextmanager
	def suppress_stdout(self):
		fd = sys.stdout.fileno()

		def _redirect_stdout(to):
			sys.stdout.close()  # + implicit flush()
			os.dup2(to.fileno(), fd)  # fd writes to 'to' file
			sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

		with os.fdopen(os.dup(fd), "w") as old_stdout:
			with open(os.devnull, "w") as file:
				_redirect_stdout(to=file)
			try:
				yield  # allow code to be run with the redirected stdout
			finally:
				_redirect_stdout(to=old_stdout)  # restore stdout.
				# buffering and flags such as
				# CLOEXEC may be different
