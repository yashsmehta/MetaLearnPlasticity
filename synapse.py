from utils import generate_gaussian
import jax.numpy as jnp
import numpy as np


def init_volterra(init=None, random_key=None):
    """Create a Volterra-parameterized synapse.

    Args:

        init (``str``):

            How to initialize the Voltera coefficients. Possible choices:
                * ``None`` (default): initialize all coefficients with zero
                * 'oja': initialize with Oja's rule
                * 'random': initialize with normally distributed random values.
                  In this case, ``random_key`` has to be provided

    Returns:

        (parameters, function)

            parameters: tensor with Volterra coefficients
            function: a function
                f(pre, post, weight, parameters) -> delta_weight
    """

    match init:

        case None:

            parameters = np.zeros((3, 3, 3))

        case 'oja':

            parameters = np.zeros((3, 3, 3))
            parameters[1][1][0] = 1
            parameters[0][2][1] = -1

        case 'random':

            assert random_key is not None, \
                "For random initialization, a random key has to be given"
            parameters = generate_gaussian(random_key, (3, 3, 3), scale=1e-5)

        case _:

            raise RuntimeError(f"init method {init} not implemented")

    return jnp.array(parameters), volterra_plasticity_function


def volterra_synapse_tensor(pre, post, weight):
    synapse_tensor = jnp.outer(
        jnp.outer(
            jnp.array([pre**0, pre**1, pre**2]),
            jnp.array([post**0, post**1, post**2])),
        jnp.array([weight**0, weight**1, weight**2]))

    synapse_tensor = jnp.reshape(synapse_tensor, (3, 3, 3))
    return synapse_tensor


def volterra_plasticity_function(pre, post, weight, volterra_coefficients):

    synapse_tensor = volterra_synapse_tensor(pre, post, weight)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw
