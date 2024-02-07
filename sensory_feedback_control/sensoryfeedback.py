"""
Created on Tue Feb 6, 2024
@author: tixianw2
"""
import numpy as np
from sensory_feedback_control.utils import _aver, _aver_kernel


class SensoryFeedback:
	def __init__(self, env, callback_list: dict, step_skip: int, ramp_up_time=0.0):
		self.env = env
		self.callback_list = callback_list
		self.every = step_skip
		self.s = env.arm_param['s']
		self.ctrl_mag = np.zeros([3, env.arm_param['n_elem']+1])
		assert ramp_up_time >= 0.0
		if ramp_up_time == 0:
			self.ramp_up_time = 1e-14
		else:
			self.ramp_up_time = ramp_up_time
		self.count = 0

	def sensor(self, system, target):
		target_vector = target[:,None] - system.position_collection[:-1,:]
		self.dist = np.sqrt(np.einsum('in,in->n', target_vector, target_vector))
		self.min_idx = np.argmin(self.dist) # -20 # -1 # 
		self.s0 = self.s[self.min_idx]
		norm = _aver(self.dist)
		norm[norm==0] += 1e-16
		normalized_target_vector = _aver(target_vector) / norm
		tangent_vector = system.tangents[:-1,:]
		normal_vector = tangent_vector.copy()
		normal_vector[0, :] = -tangent_vector[1, :]
		normal_vector[1, :] = tangent_vector[0, :]
		self.sin_alpha = np.einsum('in,in->n', normalized_target_vector, normal_vector)

	def LM_choice(self, array):
		return np.where(array>=0), np.where(array<0)
	
	def TM_choice(self, array):
		return np.where(abs(array)<=1.) # 0.1
	
	def feedback(self, time, system):
		# mag_cos = system.director_collection.copy()[1,1,:]
		# mag_sin = system.director_collection.copy()[2,1,:]
		error_feedback = self.sin_alpha # mag_cos # mag_cos - mag_sin # 0.5 * (np.sqrt(3)*mag_cos - mag_sin) # 
		idx_top, idx_bottom = self.LM_choice(error_feedback)
		# idx_central = self.TM_choice(idx_top, idx_bottom)
		sigma = 0.01 # 0.01
		# mag = 0. # 1.0 # 0.2 ## w/o transverse muscle
		steep = 300 # 300 # 200
		shift = 1.5 # 1.5 # 2.5
		## Ramp up the muscle torque
		factor = min(1.0, time / self.ramp_up_time)
		## calculating activation profile
		# # self.s0 = self.s[30]
		# # idx_top = idx_top[0][idx_top[0]<self.min_idx-2]
		# # idx_bottom = idx_bottom[0][idx_bottom[0]<self.min_idx-2]
		if self.s0 > self.s[10]:
			## longitudinal
			self.ctrl_mag[0,idx_top] = (-1 / (1 + np.exp(-steep * (self.s[idx_top] - (self.s0-shift*sigma)))) + 1) # gaussian(s, s0-4*sigma, sigma) * mag # 2.5
			self.ctrl_mag[1,idx_bottom] = (-1 / (1 + np.exp(-steep * (self.s[idx_bottom] - (self.s0-shift*sigma)))) + 1) # gaussian(s, s0+2.5*sigma, sigma) * mag
			## transverse
			# self.ctrl_mag[-1,:] = (-1 / (1 + np.exp(-steep * (self.s[:] - (self.s0-shift*sigma)))) + 1) * mag
			# # if time > 2.:
			# # self.ctrl_mag[-1,idx_central:] = (-1 / (1 + np.exp(-200 * (self.s[idx_central:] - (self.s0-2.5*sigma)))) + 1) * mag
		else:
			self.ctrl_mag[0,idx_top] = 1.
			self.ctrl_mag[1,idx_bottom] = 1.
		
		mag_feedback = _aver_kernel(abs(error_feedback))
		mag_feedback[0] = mag_feedback[1]
		mag_feedback[1] = 0.5 * (mag_feedback[0] + mag_feedback[2])
		# # mag_cos = 0.5 * (np.tanh(10*mag_feedback) + 1)
		# # mag_ramp = 1 / (1 + np.exp(-500 * (self.s-self.s[idx_central]-0.01)))
		self.ctrl_mag[:-1, :] *= factor * mag_feedback # * mag_ramp
		self.ctrl_mag = np.clip(self.ctrl_mag, 0, 1)
		# self.ctrl_mag[-1, :] *= factor * (1 - mag_feedback**2)
	
	def sensory_feedback_law(self, time, system, target):
		self.sensor(system, target)
		self.feedback(time, system)
		self.callback()
		self.count += 1
		return self.ctrl_mag
	
	def callback(self):
		if self.count % self.every == 0:
			# self.callback_list['u'].append(self.ctrl_mag.copy())
			self.callback_list['dist'].append(self.dist.copy())
			self.callback_list['angle'].append(self.sin_alpha.copy())
			self.callback_list['s_bar'].append(self.min_idx)