import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np


def generate_input_parameters(key, input_dim, num_odors, firing_fraction):
    """
    return the mus and sigmas tensors, with some mean, and variance
    """
    mus = np.zeros((num_odors, input_dim))
    mus += 1
    mask = jax.random.choice(
        key,
        np.array([0, 1]),
        shape=(num_odors, input_dim),
        p=np.array([1 - firing_fraction, firing_fraction]),
    )
    mus = np.multiply(mus, mask)
    diag_mask = np.ma.diag(np.ones(input_dim))

    sigmas = np.zeros((num_odors, input_dim, input_dim))
    firing_covariance = 0.1
    base_noise = 0.1
    sigmas += base_noise

    for i in range(num_odors):
        sigmas[i] = (firing_covariance - base_noise) * mus[i] + sigmas[i]
        sigmas[i] = np.multiply(sigmas[i], diag_mask)

    return jnp.array(mus), jnp.array(sigmas)


def generate_binary_input_parameters():
    """
    return mus, sigmas for binary encoding of 2 odors
    """
    mus = jnp.identity(2)
    sigmas = jnp.zeros((2, 2, 2))
    return mus, sigmas


def sample_inputs(key, mus, sigmas, k, scale=0.1):

    # get a normally distributed variable
    x = scale * jax.random.normal(key, shape=mus[0].shape)

    # shift and scale according to mus[k]
    x = x @ sigmas[k]
    x = x + mus[k]

    return x


def generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor):
    vsample_inputs = vmap(sample_inputs, in_axes=(None, None, 0, 0))
    vvsample_inputs = vmap(vsample_inputs, in_axes=(None, None, 1, 1))
    input_data = vvsample_inputs(mus, sigmas, odors_tensor, keys_tensor)
    input_data = jnp.swapaxes(input_data, 0, 1)
    return input_data
