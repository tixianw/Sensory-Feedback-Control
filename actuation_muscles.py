import numpy as np
from numba import njit
from elastica._linalg import _batch_matvec, _batch_cross
from elastica._calculus import quadrature_kernel, difference_kernel
from tools import _diff, _aver, _diff_kernel, _aver_kernel

# @njit(cache=True)
# def _passive_force(internal_forces, max_force, sigma):
# 	nu1 = sigma[-1, :] + 1
# 	blocksize = nu1.shape[0]
# 	internal_TM = np.zeros((3, blocksize))
# 	idx = np.where(nu1 < 1)[0]
# 	# print(idx)
# 	for i in range(len(idx)):
# 		internal_TM[2, idx[i]] = max_force[-1, idx[i]] * (1 / nu1[idx[i]] - 1) * 0.2
# 	internal_forces[:, :] += internal_TM
# 	nu1 = sigma[-1, :] + 1
# 	return

@njit(cache=True)
def _passive_force(internal_forces, S, sigma):
	blocksize = sigma.shape[-1]
	nolinear_passive = np.zeros((3, blocksize))
	for i in range(blocksize):
		nolinear_passive[2, i] = sigma[-1, i] * abs(sigma[-1, i]) * S[-1,-1,i] * 5
	internal_forces[:, :] += nolinear_passive
	return

@njit(cache=True)
def _passive_couple(internal_couples, B, kappa):
	blocksize = kappa.shape[-1]
	nolinear_passive = np.zeros((3, blocksize))
	for i in range(blocksize):
		nolinear_passive[0, i] = kappa[i] * abs(kappa[i]) * B[0,0,i] * 5
	internal_couples[:, :] += nolinear_passive
	return

@njit(cache=True)
def _row_sum(array_collection):
	rowsize = array_collection.shape[0]
	array_sum = np.zeros(array_collection.shape[1:])
	for n in range(rowsize):
		array_sum += array_collection[n, ...]
	return array_sum

@njit(cache=True)
def _material_to_lab(director_collection, vectors):
	blocksize = vectors.shape[1]
	lab_frame_vectors = np.zeros((3, blocksize))
	for n in range(blocksize):
		for i in range(3):
			for j in range(3):
				lab_frame_vectors[i, n] += (
					director_collection[j, i, n] * vectors[j, n]
				)
	return lab_frame_vectors

@njit(cache=True)
def _lab_to_material(director_collection, vectors):
	return _batch_matvec(director_collection, vectors)

@njit(cache=True)
def _internal_to_external_load(
	director_collection, tangents,
	rest_lengths, dilatation, sigma, kappa,
	shear_matrix, bend_matrix, max_force,
	internal_forces, internal_couples,
	external_forces, external_couples,
	muscle_bend_couple, muscle_shear_couple
	):
	# # rotation = director_collection.copy()[1:, 1, :]
	# # external_forces[:2, 1:-1] = np.diff(rotation * internal_forces[2, :])
	# # external_forces[:2, 0] = (rotation * internal_forces[2, :])[:, 0]
	# # external_forces[:2, -1] = -(rotation * internal_forces[2, :])[:, -1]

	# # external_couples[0, :] = np.diff(internal_couples[0, :]) + sigma[1, :] * internal_forces[2, :] * rest_lengths

	# _passive_force(internal_forces, max_force, sigma)
	_passive_force(internal_forces, shear_matrix, sigma)
	# _passive_couple(internal_couples, bend_matrix, kappa)
	
	external_forces[:, :] = difference_kernel(
		_material_to_lab(director_collection, internal_forces)
		)
	
	muscle_bend_couple[:,:] = _diff(internal_couples)
	muscle_shear_couple[:,:] = _batch_cross(
				_lab_to_material(director_collection, tangents * dilatation),
				internal_forces
			)  * rest_lengths

	external_couples[:, :] = muscle_bend_couple + muscle_shear_couple

	# external_couples[:, :] = (
	# 	_diff(internal_couples) +
	# 	# quadrature_kernel(
	# 	# 	_batch_cross(kappa, internal_couples) * rest_voronoi_lengths
	# 	# ) +
	# 	_batch_cross(
	# 			_lab_to_material(director_collection, tangents * dilatation),
	# 			internal_forces
	# 		)  * rest_lengths
	# 	)
	return

@njit(cache=True)
def longitudinal_muscle_function(
	magnitude_force_mean, magnitude_force, 
	r_LM, max_force,# radius_ratio, radius,
	director_collection, tangents, rest_lengths, 
	dilatation, sigma, kappa, 
	shear_matrix, bend_matrix, 
	# internal_forces_mean,
	internal_forces, internal_couples,
	external_forces, external_couples,
	muscle_bend_couple, muscle_shear_couple
):
	# internal_forces_mean[2, :] = magnitude_force_mean.copy()
	internal_forces[2, :] = _row_sum(magnitude_force)
	# r_m = r_LM # _aver_kernel(r_LM)
	# # internal_couples[:, :] = _batch_cross(
	# # 	r_m, 
	# # 	internal_forces_mean
	# # )
	internal_couples[0, :] = _row_sum(r_LM * magnitude_force_mean)

	_internal_to_external_load(
		director_collection, tangents,
		rest_lengths, dilatation, sigma, kappa, 
		shear_matrix, bend_matrix, max_force,
		internal_forces, internal_couples,
		external_forces, external_couples,
		muscle_bend_couple, muscle_shear_couple
	)

# force-length curve (x) = 3.06 x^3 - 13.64 x^2 + 18.01 x - 6.44
@njit(cache=True)
def force_length_curve_poly(stretch, f_l_coefficients=np.array([-57.6/39.2, 176/39.2, -80/39.2])): # np.array([-6.44, 18.01, -13.64, 3.06])):
	degree = f_l_coefficients.shape[0]
	blocksize = stretch.shape # [0] # 3 \times n_elem+1
	force_weight = np.zeros(blocksize)
	for k in range(blocksize[0]):
		for i in range(blocksize[1]):
			for power in range(degree):
				force_weight[k,i] += f_l_coefficients[power] * (stretch[k,i] ** power)
			force_weight[k,i] = 0 if force_weight[k,i] < 0 else force_weight[k,i]
	return force_weight

class ContinuousActuation:
	def __init__(self, n_elements: int, n_muscle,max_force,radius_ratio,passive_ratio):
		# self.internal_forces_mean = np.zeros((3, n_elements+1))
		self.internal_forces = np.zeros((3, n_elements)) 	   # material frame
		self.external_forces = np.zeros((3, n_elements+1))     # global frame
		self.internal_couples = np.zeros((3, n_elements+1))    # material frame
		self.external_couples = np.zeros((3, n_elements))      # material frame
		self.muscle_bend_couple = np.zeros([3, n_elements])
		self.muscle_shear_couple = np.zeros([3, n_elements])
		self.magnitude = np.zeros([n_muscle, n_elements+1])
		self.max_force = max_force
		self.muscle_radius_ratio = np.ones(n_elements) * radius_ratio[:, None]
		self.passive_ratio = passive_ratio
		self.flag_longitudinal = radius_ratio * radius_ratio
	
	def sigma_to_stretch(self, sigma):
		array = _aver_kernel(sigma[-1,:]) + 1.
		return array
	
	def kappa_to_curvature(self, kappa, dilatation):
		array = np.zeros(kappa.shape[1]+2)
		array[1:-1] = -kappa[0,:] / dilatation
		return array
	
	@staticmethod
	@njit(cache=True)
	def muscle_stretch(shape, strain, curvature, off_center_displacement):
		array = np.zeros(shape)
		array[:-1, :] += strain
		array[-1, :] += 2. - strain
		array -= off_center_displacement * curvature
		return array
	
	def set_control(self, u):
		self.magnitude *= 0
		self.magnitude = u.copy()
	
	def calculate_force_length_weight(self, system):
		self.weight = np.ones(self.magnitude.shape)
		self.nu_muscle = self.muscle_stretch(
			shape=self.weight.shape,
			strain=self.sigma_to_stretch(system.sigma),
			curvature=self.kappa_to_curvature(system.kappa, system.voronoi_dilatation),
			off_center_displacement=_aver_kernel(self.muscle_radius_ratio * system.radius),
		)
		# self.weight[...] = force_length_curve_poly(self.nu_muscle)
		
	def __call__(self, system):
		self.calculate_force_length_weight(system)
		longitudinal_muscle_function(
			self.magnitude * _aver_kernel(self.max_force) * self.weight,
			_aver(self.magnitude) * self.max_force * self.passive_ratio * _aver(self.weight), 
			_aver_kernel(self.muscle_radius_ratio * system.radius), self.max_force,
			system.director_collection, system.tangents, system.rest_lengths, 
			system.dilatation, system.sigma, system.kappa[0, :] / system.voronoi_dilatation, 
			system.shear_matrix, system.bend_matrix, 
			# self.internal_forces_mean, 
			self.internal_forces, self.internal_couples,
			self.external_forces, self.external_couples,
			self.muscle_bend_couple, self.muscle_shear_couple
		)