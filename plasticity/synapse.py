from plasticity.utils import generate_gaussian
import jax
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


def mlp_forward_pass(mlp_params, inputs):
    """Forward pass for the MLP
    Args:
        mlp_params (list): list of tuples (weights, biases) for each layer
        inputs (array): input data
    Returns:
        array: output of the network
    """
    activation = inputs
    for w, b in mlp_params[:-1]:  # for all but the last layer
        activation = jax.nn.leaky_relu(jnp.dot(activation, w) + b)
    final_w, final_b = mlp_params[-1]  # for the last layer
    logits = jnp.dot(activation, final_w) + final_b
    output = jnp.tanh(logits)
    return jnp.squeeze(output)


def mlp_plasticity_function(x, y, z, mlp_params):
    # dw is foward pass of mlp
    inputs = jnp.array([x, y, z])
    dw = mlp_forward_pass(mlp_params, inputs)
    return dw


def init_zeros():
    return np.zeros((3, 3, 3))


def init_random(key):
    assert key is not None, "For random initialization, a random key has to be given"
    return generate_gaussian(key, (3, 3, 3), scale=1e-5)


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


def init_volterra(key=None, init=None):
    init_functions = {
        "zeros": init_zeros,
        "random": lambda: init_random(key),
        "reward": lambda: init_reward(np.zeros((3, 3, 3))),
        "reward-with-decay": lambda: init_reward_with_decay(np.zeros((3, 3, 3))),
        "custom": lambda: init_custom(np.zeros((3, 3, 3))),
    }

    if init not in init_functions:
        raise RuntimeError(f"init method {init} not implemented")

    parameters = init_functions[init]()
    return jnp.array(parameters), volterra_plasticity_function


def init_plasticity_mlp(key, layer_sizes, scale=0.01):
    assert (
        layer_sizes[-1] == 1 and layer_sizes[0] == 3
    ), "output dim should be 1, and input dim should be 3"

    mlp_params = [
        (
            generate_gaussian(key, (m, n), scale),
            generate_gaussian(key, (n,), scale),
        )
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    return mlp_params, mlp_plasticity_function


def init_plasticity(key, cfg, mode):
    if "generation" in mode:
        if cfg.generation_model == "volterra":
            return init_volterra(key, init=cfg.generation_coeff_init)
        elif cfg.generation_model == "mlp":
            assert (
                cfg.generation_coeff_init == "random"
            ), "only random init supported for mlp"
            return init_plasticity_mlp(key, cfg.meta_mlp_layer_sizes)
    elif "plasticity" in mode:
        if cfg.plasticity_model == "volterra":
            return init_volterra(key, init=cfg.plasticity_coeff_init)
        elif cfg.plasticity_model == "mlp":
            assert (
                cfg.plasticity_coeff_init == "random"
            ), "only random init supported for mlp"
            return init_plasticity_mlp(key, cfg.meta_mlp_layer_sizes)

    raise RuntimeError(
        f"mode needs to be either generation or plasticity, and plasticity_model needs to be either volterra or mlp"
    )
