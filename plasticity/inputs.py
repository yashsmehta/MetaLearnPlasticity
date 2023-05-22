import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np


def generate_input_parameters(key, input_dim, num_odors, firing_fraction):
    """
    return the mus and sigmas tensors, with some mean, and variance
    """
    mus = np.zeros((num_odors, input_dim))
    mus += 0.1
    mask = jax.random.choice(
        key,
        np.array([0, 1]),
        shape=(num_odors, input_dim),
        p=np.array([1 - firing_fraction, firing_fraction]),
    )
    mus = np.multiply(mus, mask)
    diag_mask = np.ma.diag(np.ones(input_dim))

    sigmas = np.zeros((num_odors, input_dim, input_dim))
    firing_covariance = 0.01
    base_noise = 0.01
    sigmas += base_noise

    for i in range(num_odors):
        sigmas[i] = (firing_covariance - base_noise) * mus[i] + sigmas[i]
        sigmas[i] = np.multiply(sigmas[i], diag_mask)

    return jnp.array(mus), jnp.array(sigmas)

def sample_inputs(mus, sigmas, k, random_key):

    # get a normally distributed variable
    x = jax.random.normal(random_key, shape=mus[0].shape)

    # shift and scale according to mus[k]
    x = x @ sigmas[k]
    x = x + mus[k]

    return x

def get_reward_prob_tensor(odors_tensor, reward_ratios_seq):
    """
    TODO: make efficient by changing for-loop to vmap
    """
    num_trajec, len_trajec = odors_tensor.shape
    block_interval_length = len_trajec // len(reward_ratios_seq)
    reward_prob_tensor = np.zeros((num_trajec, len_trajec))

    for trial in range(num_trajec):
        for time_step in range(len_trajec):
            block = time_step // block_interval_length
            r1, r2 = reward_ratios_seq[block]
            reward_prob_tensor[trial, time_step] = r1 if odors_tensor[trial, time_step] == 0 else r2

    return reward_prob_tensor

def generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor):
    vsample_inputs = vmap(sample_inputs, in_axes=(None, None, 0, 0))
    vvsample_inputs = vmap(vsample_inputs, in_axes=(None, None, 1, 1))
    input_data = vvsample_inputs(mus, sigmas, odors_tensor, keys_tensor)
    input_data = jnp.swapaxes(input_data, 0, 1)
    return input_data
