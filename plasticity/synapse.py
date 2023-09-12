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


def init_zeros(num_rules):
    return np.zeros((num_rules, 3, 3, 3))


def init_random(key, num_rules):
    assert (
        key is not None
    ), "For random initialization, a random key has to be given"
    return generate_gaussian(key, (num_rules, 3, 3, 3), scale=1e-5)


def init_reward(parameters):
    num_rules = parameters.shape[0]
    # create a uniformly decaying array from 1 to 0 in numpy of length num_rules
    # and multiply it with the parameters
    lr_multiplier = np.linspace(1, 0, num_rules)
    parameters[:,1,1,0] = 1.
    np.einsum('i,ijkl->ijkl', lr_multiplier, parameters, out=parameters)
    return parameters


def init_volterra(key, num_rules, init=None):
    init_functions = {
        "zeros": lambda: init_zeros(num_rules),
        "random": lambda: init_random(key, num_rules),
        "reward": lambda: init_reward(np.zeros((num_rules, 3, 3, 3))),
    }

    if init not in init_functions:
        raise RuntimeError(f"init method {init} not implemented")

    parameters = init_functions[init]()
    return jnp.array(parameters), volterra_plasticity_function
