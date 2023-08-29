from plasticity.behavior.utils import generate_gaussian
import jax.numpy as jnp
import numpy as np


def volterra_synapse_tensor(x, y, z):

    synapse_tensor = jnp.outer(
        jnp.outer(
            jnp.array([x**0, x**1, x**2]),
            jnp.array([y**0, y**1, y**2]),
        ),
        jnp.array([z**0, z**1, z**2]),
    )
    synapse_tensor = jnp.reshape(synapse_tensor, (3, 3, 3))
    return synapse_tensor


def volterra_plasticity_function(x, y, z, volterra_coefficients):
    synapse_tensor = volterra_synapse_tensor(x, y, z)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw


def init_zeros():
    return np.zeros((3, 3, 3))


def init_random(random_key):
    assert (
        random_key is not None
    ), "For random initialization, a random key has to be given"
    return generate_gaussian(random_key, (3, 3, 3), scale=1e-5)


def init_reward(parameters):
    parameters[1][1][0] = 1
    return parameters


def init_reward_with_decay(parameters):
    parameters[1][1][0] = 0.2
    parameters[0][0][1] = -0.03
    return parameters


def init_custom(parameters):
    parameters[1][0][0] = 0.5
    parameters[0][0][1] = -0.25
    return parameters


def init_oja(parameters):
    parameters[1][1][0] = 1
    parameters[0][2][1] = -1
    return parameters


def init_volterra(random_key=None, init=None):

    init_functions = {
        "zeros": init_zeros,
        "random": lambda: init_random(random_key),
        "reward": lambda: init_reward(np.zeros((3, 3, 3))),
        "reward-with-decay": lambda: init_reward_with_decay(np.zeros((3, 3, 3))),
        "custom": lambda: init_custom(np.zeros((3, 3, 3))),
    }

    if init not in init_functions:
        raise RuntimeError(f"init method {init} not implemented")

    parameters = init_functions[init]()
    return jnp.array(parameters), volterra_plasticity_function
