"""
Created on Tue Feb 6, 2024
@author: tixianw2
"""
# import sys
# sys.path.append("../../") 
# sys.path.append("../") 
import numpy as np
from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check
from sensory_feedback_control.actuation_muscles import ContinuousActuation
from sensory_feedback_control.utils import (
	_lab_to_material, 
	_material_to_lab, 
	_aver, 
	_aver_kernel
)
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
	pass

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

class DragForce(NoForces):
	def __init__(
		self,
		n_elem,
		rho_water,
		c_per,
		c_tan,
	):
		super(DragForce, self).__init__()
		self.rho_water = rho_water
		self.c_per = c_per
		self.c_tan = c_tan
		self.drag_force = np.zeros([3, n_elem+1])
		# self.count = 0

	def apply_torques(self, system, time: np.float = 0.0):
		Pa = 2 * system.radius * system.rest_lengths * system.dilatation
		Sa = Pa * np.pi
		self.velocity_material_frame = _lab_to_material(system.director_collection, _aver(system.velocity_collection))
		self.force_material_frame = -self.velocity_material_frame * abs(self.velocity_material_frame)
		self.force_material_frame[-1, :] *= Sa * self.c_tan
		self.force_material_frame[1, :] *= Pa * self.c_per
		self.drag_force = _aver_kernel(_material_to_lab(system.director_collection, self.force_material_frame))
		system.external_forces += self.drag_force
		# self.callback()
		# self.count += 1

	def callback(self):
		pass

class MuscleCtrl(NoForces):
	def __init__(
		self,
		muscle,
		callback_list: dict,
		step_skip: int,
		ramp_up_time=0.0,
	):
		super(MuscleCtrl, self).__init__()
		self.muscle = muscle
		self.callback_list = callback_list
		self.every = step_skip
		assert ramp_up_time >= 0.0
		if ramp_up_time == 0:
			self.ramp_up_time = 1e-14
		else:
			self.ramp_up_time = ramp_up_time
		self.count = 0
	
	# def set_activation(self, activation):
	# 	# self.ctrl_mag = np.zeros([3, system.n_elems+1])
	# 	self.ctrl_mag = activation
	
	def apply_torques(self, system, time: np.float = 0.0):
		# self.muscle.set_control(self.ctrl_mag)
		self.muscle(system)

		system.external_forces += self.muscle.external_forces
		system.external_torques += self.muscle.external_couples
		self.callback()
		self.count += 1
	
	def callback(self):
		if self.count % self.every == 0:
			# self.callback_list['u'].append(self.ctrl_mag.copy())
			self.callback_list['u'].append(self.muscle.magnitude.copy())

class ArmEnvironment:
	def __init__(self, final_time, flags, time_step=1.0e-5, recording_fps=100):
		# Integrator type
		self.StatefulStepper = PositionVerlet()

		self.final_time = final_time
		self.time_step = time_step
		self.total_steps = int(final_time * 1e5)+1 # int(self.final_time/self.time_step) # 
		self.recording_fps = recording_fps
		self.step_skip = int(1.0 / (self.recording_fps * self.time_step))
		print('total_steps:', self.total_steps, ', final_time:', final_time)

		self.numeric_param = {
			'step_size': self.time_step, 
			'step_skip': self.step_skip,
			'final_time': self.final_time,
			'total_steps': self.total_steps,
		}
		self.flags = flags
		self.flag_shooting = flags[0]
		self.flag_target = flags[1]
		self.flag_obstacle = flags[2]
		self.flag_stop = False

	def get_systems(self,):
		return self.simulator
	
	### Set up a rod
	def set_rod(self):
		n_elem = 100 # 150 # 50 # N_exp # 
		start = np.zeros((3,))
		self.direction = np.array([1.0, 0.0, 0.0])  # np.array([0.0, 0.0, 1.0])
		self.normal = np.array([0.0, 0.0, -1.0])  # np.array([0.0, 1.0, 0.0])
		base_length = 0.2 # 0.2
		# base_radius = 0.01 # 0.025
		radius_base = base_length/20 # 0.012 # 0.01 # 0.02  # radius of the arm at the base
		radius_tip = radius_base/10 # 0.0012 # 0.004 # 0.004 # 0.0004  # radius of the arm at the tip
		radius = np.linspace(radius_base, radius_tip, n_elem+1)
		# radius = base_radius
		base_radius = (radius[:-1]+radius[1:])/2
		density = 1042 # 700 # 1000
		damping = 0.01
		# damping_torque = 0.01
		# dissipation = damping * ((base_radius/radius_base)**2) # 0.02 np.ones(len(base_radius)) # 
		# nu_torque = damping_torque * ((base_radius/radius_base)**4) # 0.01 np.ones(len(base_radius)) # 
		E = 1e4 # 1e4
		poisson_ratio = 0.5
		shear_modulus = E / 2 * (poisson_ratio + 1.0) # E/10
		## initial position and rest curvature
		# self.flag_shooting = 1
		if self.flag_shooting:
			name = 'initial_data'
			init_data = np.load('Data/'+name+'.npy', allow_pickle='TRUE').item()
			init_pos = init_data['pos']
		else:
			# if direction[0]:
			init_pos = np.vstack([np.linspace(0,base_length,n_elem+1), np.zeros([2,n_elem+1])])
			# elif direction[1]:
				# init_pos = np.vstack([np.zeros(n_elem+1), -np.linspace(0,base_length,n_elem+1), np.zeros(n_elem+1)])
		
		s = np.linspace(0, base_length, n_elem+1)
		# s_mean = (s[1:] + s[:-1])/2

		self.arm_param = {'n_elem': n_elem,
			'L': base_length,
			'radius': radius,
			'base_radius': base_radius,
			'rho': density,
			'damping': damping,
			# 'damping_torque': damping_torque,
			# 'dissipation': dissipation,
			# 'nu_torque': nu_torque,
			'E': E,
			'G': shear_modulus, 
			'initial_pos': init_pos,
			's': s,
			}
		
		self.shearable_rod = CosseratRod.straight_rod(
			n_elem,
			start,
			self.direction,
			self.normal,
			base_length,
			base_radius,
			density,
			0.0, # dissipation,
			youngs_modulus=E,
			# nu_for_torques=nu_torque,
			shear_modulus=shear_modulus,
			position=init_pos,
		)
		self.shearable_rod.shear_matrix *= 1000 ## inextensible + unshearable

		self.simulator.append(self.shearable_rod)
		### Add analytic damping
		self.simulator.dampen(self.shearable_rod).using(
			AnalyticalLinearDamper,
			damping_constant=damping/density/(np.pi*radius_base**2)/5, # /2,
			time_step=self.time_step,
		)
		## Add Constraints
		self.simulator.constrain(self.shearable_rod).using(
			OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
		)
		# self.simulator.connect(self.shearable_rod, self.shearable_rod).using(
		# 	SelfContact, k=1.0, nu=0.1,
		# )

	def add_drag(self):
		### environment parameters
		fluid_factor = 5 # 5 # 0
		gravitational_acc = -9.80665 * fluid_factor * 0
		rho_water = 1022
		c_per = 0.5 * rho_water * 2 * fluid_factor # 1.013
		c_tan = 0.5 * rho_water * 0.01 * fluid_factor # 0.0256
		
		# ## Add gravitational forces
		# self.simulator.add_forcing_to(self.shearable_rod).using(
		# 	GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
		# )
		## Add fluid drag forces
		self.simulator.add_forcing_to(self.shearable_rod).using(
			DragForce,
			self.shearable_rod.n_elems,
			rho_water,
			c_per,
			c_tan=c_tan,
		)
	
	def add_muscles(self):
		### multiple muscle fibers
		n_max = np.array([1., 1, -2]) * 1. # 2. # 1.5 # 4
		n_muscle = len(n_max) # number of muscle fiber
		max_force = (self.shearable_rod.radius/self.shearable_rod.radius[0])**2 * n_max[:, None]
		passive_ratio = 1
		radius_ratio = np.array([1, -1, 0])
		# # print(np.count_nonzero(radius_ratio),'longitudinal, ', len(radius_ratio)-np.count_nonzero(radius_ratio), 'transverse')
		# print('n_max:', n_max)

		self.muscle = ContinuousActuation(
			self.shearable_rod.n_elems,
			n_muscle=n_muscle,
			max_force=max_force,
			radius_ratio=radius_ratio,
			passive_ratio=passive_ratio,
		)

		self.muscle_list = defaultdict(list)
		self.simulator.add_forcing_to(self.shearable_rod).using(
			MuscleCtrl,
			self.muscle,
			self.muscle_list,
			self.step_skip,
			ramp_up_time=0.01 # 0.2 # 0.01 # 0.8 # 0.5 # 0.1
		)

	def set_arm(self):
		self.set_rod()
		self.add_muscles()
		self.add_drag()
		self.callback()
	
	def setup(self):
		## Set up a rod
		self.set_arm()
	
	def callback(self):
		self.pp_list = defaultdict(list)
		self.simulator.collect_diagnostics(self.shearable_rod).using(
			CallBack, 
			step_skip=self.step_skip, 
			callback_params=self.pp_list, 
			total_steps=self.total_steps,
		)
	
	def reset(self):
		self.simulator = BaseSimulator()

		self.setup()

		# Finalize the simulator and create time stepper
		self.simulator.finalize()
		self.do_step, self.stages_and_updates = extend_stepper_interface(
			self.StatefulStepper, self.simulator
		)

		""" Return 
			(1) total time steps for the simulation step iterations
			(2) plant systems
		"""
		return self.total_steps, self.get_systems()

	def step(self, time, activation):

		self.muscle.set_control(activation)
		
		# Run the simulation for one step
		time = self.do_step(
			self.StatefulStepper,
			self.stages_and_updates,
			self.simulator,
			time,
			self.time_step,
		)

		# Done is a boolean to reset the environment before episode is completed
		done = False
		# Position of the rod cannot be NaN, it is not valid, stop the simulation
		invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

		if invalid_values_condition == True:
			print("NaN detected in the simulation !!!!!!!!")
			done = True
		
		if self.flag_stop:
			done = True

		""" Return
			(1) current simulation time
			(2) current systems
			(3) a flag denotes whether the simulation runs correlectly
		"""
		return time, self.get_systems(), done

