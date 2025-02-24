"""
Created on Tue Feb 6, 2024
@author: tixianw2
"""
# import numpy as np
# from elastica import *
from set_arm_environment import *
# from sensory_feedback_control.utils import target_traj

# Add call backs
class CallBack(CallBackBaseClass):
	"""
	Call back function for the arm
	"""
	def __init__(self, step_skip: int, callback_params: dict, total_steps: int):
		CallBackBaseClass.__init__(self)
		self.every = step_skip
		self.callback_params = callback_params
		self.total_steps = total_steps
		self.count = 0

	def make_callback(self, system, time, current_step: int):
		self.callback_params['potential_energy'].append(
			system.compute_bending_energy() + system.compute_shear_energy()
		)
		if self.count % self.every == 0: # and self.count != self.total_steps:
			self.callback_params["time"].append(time)
			# self.callback_params["step"].append(current_step)
			self.callback_params["position"].append(
				system.position_collection.copy()
			)
			self.callback_params["orientation"].append(
				system.director_collection.copy()
			)
			self.callback_params["velocity"].append(
				system.velocity_collection.copy()
			)
			self.callback_params["angular_velocity"].append(
				system.omega_collection.copy()
			)
			
			self.callback_params["kappa"].append(
				system.kappa.copy()
			)
			self.callback_params['strain'].append(
				system.sigma.copy() # + np.array([0, 0, 1])[:, None]
			)
			self.callback_params['dilatation'].append(
				system.dilatation.copy()
			)
			self.callback_params['voronoi_dilatation'].append(
				system.voronoi_dilatation.copy()
			)
			
		self.count += 1
		return

class Environment(ArmEnvironment):

	def setup(self):
		## Set up a rod
		self.set_arm()
		## Set up the target
		if self.flag_target:
			self.set_target()
		if self.flag_obstacle:
			self.set_obstacle()
		return
		
	def set_target(self):
		if not self.flag_shooting:
			x_star = 0.1
			y_star = 0.1 # 0.15
		else:
			x_star = 0.2 # 0.15
			y_star = 0.1
		print('target:', np.array([x_star,y_star]))

		self.target = np.zeros([self.total_steps, 2])
		
		# ## moving target
		# pos1 = np.array([x_star, y_star])
		# pos2 = np.array([-x_star*1., y_star])
		# # pos2 = np.array([x_star*1.5, -y_star])
		# # target_traj(target, pos1, pos1, 0., 0.15, self.total_steps)
		# target_traj(self.target, pos1, pos2, 0., 1., self.total_steps)
		# # target_traj(target, pos2, pos2, 0.8, 1., self.total_steps)

		## static target
		self.target[:,:] = np.array([x_star,y_star])

	def set_obstacle(self):
		N_obs = 2
		r_obs = np.ones([N_obs]) * 0.02 # 0.01 # 0.04
		len_obs = np.ones([N_obs]) * 0.04
		pos_obs = [np.array([0.065,0.05,0]), np.array([0.14,0.05,0])] # [np.array([0.1,0.05,0])] # [np.array([0.,0.03,0]), np.array([0.08,0.03,0])] # [np.hstack([target_exp+0.01,0])] # 
		# [np.array([0.065,0.05,0]), np.array([0.135,0.05,0])] # 
		self.obstacle_param = {
			'N_obs': N_obs, 
			'r_obs': r_obs, 
			'pos_obs': np.array(pos_obs), 
			'len_obs': len_obs,
			}
		name_obstacle = locals()
		for o in range(N_obs):
			name_obstacle['obstacle'+str(o)] = Cylinder(
				start=pos_obs[o],
				direction=self.normal,
				normal=self.direction,
				base_length=len_obs[o],
				base_radius=r_obs[o],
				density=500,
			)
			self.simulator.append(name_obstacle['obstacle'+str(o)])
			self.simulator.constrain(name_obstacle['obstacle'+str(o)]).using(
				OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
			)
			self.simulator.connect(self.shearable_rod, name_obstacle['obstacle'+str(o)]).using(
				ExternalContact, k=1e2, nu=1.0
			)
	
	def callback(self):
		self.pp_list = defaultdict(list)
		self.simulator.collect_diagnostics(self.shearable_rod).using(
			CallBack, step_skip=self.step_skip, 
			callback_params=self.pp_list, total_steps=self.total_steps,
		)

