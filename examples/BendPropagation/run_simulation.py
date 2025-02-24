"""
Created on Tue Feb 6, 2024
@author: tixianw2
"""
import sys
sys.path.append("../../") 
import numpy as np
from tqdm import tqdm
from elastica import *
# from set_arm_environment import ArmEnvironment
from sensory_feedback_control.sensoryfeedback import SensoryFeedback
from set_environment import Environment

folder = '../'

def data_save(env, controller):
	model = {
		'flags': env.flags,
		'numerics': env.numeric_param,
		'arm': env.arm_param,
	}
	if env.flags[1]:
		model.update({'target': env.target[::env.step_skip, :]})
	if env.flags[2]:
		model.update({'obstacle': env.obstacle_param})
	position = []
	arm = []
	muscle = []
	sensor = []
	position.append(np.array(env.pp_list['position'])[:,:2,:])
	arm.append({
		'orientation': np.array(env.pp_list['orientation'])[:,:,:,:],
		'velocity': np.array(env.pp_list['velocity'])[:,:2,:],
		'omega': np.array(env.pp_list['angular_velocity'])[:,0,:],
		'kappa': -np.array(env.pp_list['kappa'])[:,0,:] / np.array(env.pp_list['voronoi_dilatation'])[:,:],
		'nu1': np.array(env.pp_list['strain'])[:, -1, :] + 1, 
		'nu2': np.array(env.pp_list['strain'])[:, 1, :],
		# 'e': np.array(env.pp_list['dilatation'])[:,:],
		# 'v_e': np.array(env.pp_list['voronoi_dilatation'])[:, :],
		# 'g': np.array(env.pp_list['shear_force'])[:, :],
	})
	muscle.append({
		'u': np.array(env.muscle_list['u']),
	})
	sensor.append({
		'dist': np.array(controller.callback_list['dist']),
		'angle': np.array(controller.callback_list['angle']),
		's_bar': np.array(controller.callback_list['s_bar']),
		})

	data = {'t': np.array(env.pp_list['time']),
		'model': model,
		'position': position,
		'arm': arm,
		'muscle': muscle,
		'sensor': sensor,
		}
	np.save(folder+'Data/test2.npy', data)

def get_activation(systems, time, controller=None, target=None):
	if controller==None:
		activation = np.zeros([3,systems[0].n_elems+1])
	else:
		activation = controller.sensory_feedback_law(time, systems[0], target)
	return activation


def main(filename):
	## Create arm and simulation environment
	final_time = 2.0 # 1.5 # 1.001
	flag_shooting = 1 # initialize the arm with given bent profile
	flag_target = True # False # 
	flag_obstacle = False # True # 
	flags = [flag_shooting, flag_target, flag_obstacle]

	# env = ArmEnvironment(final_time, flags)
	env = Environment(final_time, flags)
	total_steps, systems = env.reset()

	## Create sensory feedback controller
	sensor_list = defaultdict(list)
	controller = SensoryFeedback(env, sensor_list, env.step_skip)
	
	# Start the simulation
	print("Running simulation ...")
	time = np.float64(0.0)
	for k_sim in tqdm(range(total_steps)):
		target = env.target[k_sim, :]
		activation = get_activation(systems, time, controller, target)
		time, systems, done = env.step(time, activation)
		if done:
			break
	
	data_save(env, controller)
	return


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		description='Run simulation'
	)
	parser.add_argument(
		'--filename', type=str, default='simulation',
		help='a str: data file name',
	)
	args = parser.parse_args()
	main(filename=args.filename)
