## License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. 
## Applied to sections that are not already under other licenses as specified.

import os
import sys
import numpy as np
import cv2
import mujoco
from PIL import Image
np.set_printoptions(precision=4, edgeitems=10, linewidth=100)
from IPython.display import display, clear_output


class MJK(object):
	output_shape = None
	name = 'mjk'

	"""docstring for MJK"""
	def __init__(self, xml_path="mjk_resources/world.xml", input_shape = (120, 120, 3), joints=None, skip_frame=None):
		self.xml_path = xml_path
		self.env = mujoco.MjModel.from_xml_path(self.xml_path)
		self.data = mujoco.MjData(self.env)
		self.input_shape = input_shape
		self.skip_frame=120

		#print(dir(mujoco)) # if you're having issues with mujoco, here's where to start debugging.
		self.renderer = mujoco.Renderer(self.env, self.input_shape[0], self.input_shape[1])
		self.renderer2 = mujoco.Renderer(self.env, 480, 480)
		self.joints = {
			"r_shoulder_pan_joint" : 
				{"low": -2.2854 , "high": 0.714602 },
			"r_shoulder_lift_joint" : 
				{"low": -0.5236 , "high": 1.3963 },
			"r_upper_arm_roll_joint" : 
				{"low": -3.9    , "high": 0.8 },
			"r_elbow_flex_joint" : 
				{"low": -2.3213 , "high": 0 },
			"r_forearm_roll_joint" : 
				{"low": -3.14   , "high": 3.14 },
			"r_wrist_flex_joint" : 
				{"low": -2.094  , "high": 0 },
			"r_wrist_roll_joint" : 
				{"low": -3.14   , "high": 3.14 },
			"r_gripper_l_finger_joint" : 
				{"low": 0       , "high": 0.548 },
			"r_gripper_l_finger_tip_joint" : 
				{"low": -0.548  , "high": 0 },
			"r_gripper_r_finger_joint" : 
				{"low": 0       , "high": 0.548 },
			"r_gripper_r_finger_tip_joint" : 
				{"low": -.548   , "high": 0 }
		} # These start at all zeros

		self.output_shape = len(self.joints)
		#self.policy = np.full((self.output_shape), 0)
		return 

	def setup(self):
		mujoco.mj_resetData(self.env, self.data)
		mujoco.mj_forward(self.env, self.data)
		for i in range(self.skip_frame):
			mujoco.mj_step(self.env, self.data)
			self.renderer.update_scene(self.data, camera="cam2")
			#cv2.namedWindow("state", cv2.WINDOW_NORMAL)
			#cv2.imshow('state', cv2.cvtColor(np.array(self.renderer.render().copy()), cv2.COLOR_BGR2RGB))
			#cv2.waitKey(1)
			#display(cv2.cvtColor(np.array(self.renderer.render().copy()), cv2.COLOR_BGR2RGB))
		self.original_distance = np.sqrt(np.sum((self.data.geom('AG8_8').xpos - 
			self.data.geom('CHEESEGcenter').xpos)**2))
		self.policy = np.random.uniform(-1.0,1.0,self.output_shape)
		self.play(self.policy)
		return

	def test_setup(self):
		images = []
		images2 = []
		mujoco.mj_resetData(self.env, self.data)
		#cv2.namedWindow("state", cv2.WINDOW_NORMAL)
		for i in range(self.skip_frame):
			mujoco.mj_step(self.env, self.data)
			self.renderer.update_scene(self.data, camera="cam2")
			self.renderer2.update_scene(self.data, camera="cam1")
			images.append(Image.fromarray(self.renderer.render().copy()))
			images2.append(Image.fromarray(self.renderer2.render().copy()))
			
			#cv2.imshow('state', cv2.cvtColor(np.array(self.renderer.render().copy()), cv2.COLOR_BGR2RGB))
			#cv2.waitKey(1)
			display(Image.fromarray(self.renderer.render().copy()).resize((300, 300)))
			clear_output(wait=True)
		self.original_distance = np.sqrt(np.sum((self.data.geom('AG8_8').xpos - 
			self.data.geom('CHEESEGcenter').xpos)**2))

		self.policy = np.random.uniform(-1.0,1.0,self.output_shape)
		self.play(self.policy)
		self.renderer.update_scene(self.data, camera="cam2")
		self.renderer2.update_scene(self.data, camera="cam1")
		images.append(Image.fromarray(self.renderer.render().copy()))
		images2.append(Image.fromarray(self.renderer2.render().copy()))
		return self.original_distance, images, images2

	def state(self):
		self.renderer.update_scene(self.data, camera="cam2")
		img = self.renderer.render().copy()
		return img

	'''
	def observation(self):
		self.renderer2.update_scene(self.data, camera="cam1")
		img = self.renderer2.render().copy()
		#cv2.imshow('observe', cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
		#cv2.waitKey(1)
		return img
	'''

	def play(self, exp_action): 
		self.policy = exp_action
		for _joint in self.joints.keys():
			action = exp_action[list(self.joints.keys()).index(_joint)] #+ self.data.joint(_joint).qpos[0]
			#legal_action = np.clip(action, self.joints[_joint]['low'], self.joints[_joint]['high'])
			#policy_new.append(legal_action)
			self.data.joint(_joint).qvel = action#legal_action
		
		mujoco.mj_step(self.env, self.data, nstep=self.output_shape)
		self.renderer.update_scene(self.data, camera="cam2")
		#cv2.imshow('state', cv2.cvtColor(np.array(self.renderer.render().copy()), cv2.COLOR_BGR2RGB))
		#cv2.waitKey(1)
		display(Image.fromarray(self.renderer.render().copy()).resize((300, 300)))
		clear_output(wait=True)
		return

	def reward(self, step):
		step += 1
		dist = np.linalg.norm(self.data.geom('AG8_8').xpos - 
			self.data.geom('CHEESEGcenter').xpos)
		reward = (self.original_distance - dist) * 100
		'''
		if ( -1.0 < reward < 0.0 ):
			reward = -1.0
		elif (0.0 < reward < 1.0 ):
			reward = 1.0
		'''
		reward = reward / step  # So the timestep is reducing the reward
		#print('	reward:	{} '.format(reward))
		return reward

	def close(self):
		cv2.destroyAllWindows()
		#self.env.close()
		return True