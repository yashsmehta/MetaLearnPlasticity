import jax
import jax.numpy as jnp
import numpy as np


def generate_input_parameters(cfg):
    """
    return the mus and sigmas tensors, with some mean, and variance
    """
    # for now, this is hardcoded to 2 odors
    num_odors = 2
    input_dim = cfg.input_dim
    firing_idx = np.random.choice(
        np.arange(input_dim), size=input_dim // 2, replace=False
    )
    mus_a = np.zeros(input_dim)
    mus_a[firing_idx] = cfg.input_firing_mean 

    mus_b = cfg.input_firing_mean * np.ones(input_dim)
    mus_b[firing_idx] = 0.
    mus = np.vstack((mus_a, mus_b))

    diag_mask = np.ma.diag(np.ones(input_dim))

    sigmas = cfg.input_noise * np.ones((num_odors, input_dim, input_dim))

    for i in range(num_odors):
        sigmas[i] = np.multiply(sigmas[i], diag_mask)

    return jnp.array(mus), jnp.array(sigmas)


def sample_inputs(key, mus, sigmas, odor):

    input_dim = mus.shape[1]
    x = jax.random.normal(key, shape=(input_dim,))

    # shift and scale according to mus[odor]
    x = x @ sigmas[odor]
    x = x + mus[odor]

    return x