import jax
import jax.numpy as jnp
import numpy as np
import sklearn.metrics
from plasticity import network, inputs

def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def generate_random_connectivity(key, m, n, sparsity):
    """
    returns a random binary connectivity mask of shape [M x N]
    with a specific connectivity sparseness
    """

    return jax.random.bernoulli(key, p=float(sparsity), shape=(m, n))


def get_r2_score(winit, connectivity_matrix, student_coefficients, student_plasticity_function, teacher_coefficients, teacher_plasticity_function):
    key = jax.random.PRNGKey(66)
    num_trajec_test = 10
    num_odors = 20
    input_dim = winit.shape[0]
    mus, sigmas = inputs.generate_input_parameters(
        key, input_dim, num_odors, firing_fraction=0.2
    )
    key, key2 = jax.random.split(key)
    odors_tensor = jax.random.choice(
        key2, jnp.arange(num_odors), shape=(num_trajec_test, 100)
    )
    keys_tensor = jax.random.split(key, num=(num_trajec_test * 100))
    keys_tensor = keys_tensor.reshape(num_trajec_test, 100, 2)
    input_data = inputs.generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor)

    teacher_trajectories = network.generate_trajectories(
        input_data,
        winit,
        connectivity_matrix,
        teacher_coefficients,
        teacher_plasticity_function,
    )
    student_trajectories = network.generate_trajectories(
        input_data,
        winit,
        connectivity_matrix,
        student_coefficients,
        student_plasticity_function,
    )
    r2_scores = np.zeros(num_trajec_test)
    for i in range(num_trajec_test):
        r2_scores[i] = sklearn.metrics.r2_score(
            teacher_trajectories[i], student_trajectories[i]
        )
    return jnp.mean(r2_scores)


def old_generate_KC_input_patterns(key, num_odors, dimensions):

    """
    each odor is represented by sparse activation of a subset of neurons (10%).
    we assume that each input KC fires with normal distribution with a different
    mean, which is sampled from 0.5 to 1 (discrete).
    KCs will have different activity means, but the same standard deviation.
    """
    num_trajec, len_trajec, input_dim = dimensions

    KC_mean_activity = jax.random.choice(
        key, jnp.arange(0.5, 1, 0.1), shape=(input_dim,)
    )
    trajectories_mean_activity = jnp.multiply(
        KC_mean_activity, jnp.ones((num_trajec, len_trajec, input_dim))
    )
    key, key2 = jax.random.split(key)
    activity_noise = jax.random.normal(key, (num_trajec, len_trajec, input_dim))
    trajectories_activity = trajectories_mean_activity + activity_noise
    odors_encoding = jax.random.bernoulli(key, p=0.2, shape=(num_odors, input_dim))
    mask = jax.random.choice(key2, odors_encoding, shape=(num_trajec, len_trajec))

    return trajectories_activity * mask
