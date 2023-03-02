import jax
import jax.numpy as jnp
import numpy as np

def generate_input_parameters(key, input_dim, num_odors, firing_fraction):
    """
    return the mus and sigmas tensors, with some mean, and variance
    """
    mus = np.zeros((num_odors, input_dim))
    mus += 1
    mask = jax.random.choice(key, np.array([0, 1]), shape=(num_odors, input_dim), p=np.array([1-firing_fraction, firing_fraction]))
    mus = np.multiply(mus, mask)
    diag_mask = np.ma.diag(np.ones(input_dim))

    sigmas = np.zeros((num_odors, input_dim, input_dim))
    firing_covariance = 0.1
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
