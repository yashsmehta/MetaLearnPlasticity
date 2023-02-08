import jax
import jax.numpy as jnp


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def generate_random_connectivity(key, neurons, sparsity):
    """
    returns a square connectivity matrix of shape [neurons x neurons]
    with a specific sparsity of connections
    """

    return jax.random.bernoulli(key, p=sparsity, shape=(neurons, neurons))


def generate_KC_input_patterns(key, num_odors, dimensions):

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
