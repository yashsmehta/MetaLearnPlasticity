from plasticity.utils import generate_gaussian
import jax.numpy as jnp
import numpy as np


def init_zeros():
    return np.zeros((3, 3, 3))

def init_random(random_key):
    assert random_key is not None, "For random initialization, a random key has to be given"
    return generate_gaussian(random_key, (3, 3, 3), scale=1e-4)

def init_reward(parameters):
    parameters[1][1][0] = 1
    return parameters

def init_reward_with_decay(parameters):
    parameters[1][1][0] = 1
    parameters[0][0][1] = -1
    return parameters

def init_custom(parameters):
    parameters[1][0][0] = 0.5
    parameters[0][0][1] = -0.25
    return parameters

def init_oja(parameters):
    parameters[1][1][0] = 1
    parameters[0][2][1] = -1
    return parameters

def init_dopamine_volterra(init=None, random_key=None):

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

def init_volterra(init=None, random_key=None):

    init_functions = {
        "zeros": init_zeros,
        "random": lambda: init_random(random_key),
        "oja": lambda: init_oja(np.zeros((3, 3, 3))),
        "custom": lambda: init_custom(np.zeros((3, 3, 3))),
    }
    
    if init not in init_functions:
        raise RuntimeError(f"init method {init} not implemented")
    
    parameters = init_functions[init]()
    return jnp.array(parameters), volterra_plasticity_function


def volterra_synapse_tensor(pre, reward_term, weight):
    """
    Note: this is the updated function incorporating the reward term,
    instad of 'y' (post synaptic activity)
    """
    synapse_tensor = jnp.outer(
        jnp.outer(
            jnp.array([pre**0, pre**1, pre**2]),
            jnp.array([reward_term**0, reward_term**1, reward_term**2]),
        ),
            jnp.array([weight**0, weight**1, weight**2]),
    )

    synapse_tensor = jnp.reshape(synapse_tensor, (3, 3, 3))
    return synapse_tensor


def volterra_plasticity_function(pre, reward_term, weight, volterra_coefficients):
    synapse_tensor = volterra_synapse_tensor(pre, reward_term, weight)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw
