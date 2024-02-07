import numpy as np
from numpy import zeros, empty
import numba
from numba import njit
from elastica._linalg import _batch_matvec, _batch_cross

## target position
def target_traj(target, start_pos, end_pos, start_time, end_time, total_steps):
    start_idx = int(total_steps*start_time)
    end_idx = int(total_steps*end_time)
    time_step = end_idx - start_idx
    target[start_idx:end_idx, :] = np.vstack([
        start_pos[0] + np.linspace(0,1,time_step)*(end_pos[0]-start_pos[0]),
        start_pos[1] + np.linspace(0,1,time_step)*(end_pos[1]-start_pos[1])
    ]).T

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
def _aver_kernel(array_collection):
    """
    Simple trapezoidal quadrature rule with zero at end-points, in a dimension agnostic way

    Parameters
    ----------
    array_collection

    Returns
    -------
    Notes
    -----
    Micro benchmark results, for a block size of 100, using timeit
    Python version: 8.14 µs ± 1.42 µs per loop
    This version: 781 ns ± 18.3 ns per loop
    """
    blocksize = array_collection.shape[-1]
    temp_collection = empty(array_collection.shape[:-1] + (blocksize + 1,))

    temp_collection[... , 0] = 0.5 * array_collection[..., 0]
    temp_collection[..., blocksize] = 0.5 * array_collection[..., blocksize - 1]

    for k in range(1, blocksize):
        temp_collection[..., k] = 0.5 * (
            array_collection[..., k] + array_collection[..., k - 1]
        )
    return temp_collection


@njit(cache=True)
def _diff_kernel(array_collection):
    """
    This function does differentiation.
    Parameters
    ----------
    array_collection

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 9.07 µs ± 2.15 µs per loop
    This version: 952 ns ± 91.1 ns per loop
    """
    blocksize = array_collection.shape[-1]
    temp_collection = empty(array_collection.shape[:-1] + (blocksize + 1,))

    temp_collection[..., 0] = array_collection[..., 0]
    temp_collection[..., blocksize] = -array_collection[..., blocksize - 1]

    for k in range(1, blocksize):
        temp_collection[..., k] = array_collection[..., k] - array_collection[..., k - 1]
    return temp_collection


@njit(cache=True)
def _diff(array_collection):
    """
    This function computes difference between elements of a batch vector
    Parameters
    ----------
    vector

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 3.29 µs ± 767 ns per loop
    This version: 840 ns ± 14.5 ns per loop
    """
    blocksize = array_collection.shape[-1]
    output_vector = empty(array_collection.shape[:-1] + (blocksize - 1,))

    for k in range(1, blocksize):
        output_vector[..., k-1] = array_collection[..., k] - array_collection[..., k-1]
    return output_vector


@njit(cache=True)
def _aver(array_collection):
    """
    This function computes the average between elements of a vector
    Parameters
    ----------
    vector

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Python version: 2.37 µs ± 764 ns per loop
    This version: 713 ns ± 3.69 ns per loop
    """
    blocksize = array_collection.shape[-1]
    output_vector = empty(array_collection.shape[:-1] + (blocksize - 1,))

    for k in range(1, blocksize):
        output_vector[..., k-1] = 0.5 * (array_collection[..., k] + array_collection[..., k-1])
    return output_vector