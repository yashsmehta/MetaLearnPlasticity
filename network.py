import jax
import jax.numpy as jnp
from jax.lax import reshape
from jax import vmap


def generate_trajectories(
        input_data,
        initial_weights,
        activation_function,
        volterra_coefficients):

    return vmap(generate_trajectory, in_axes=(0, None, None, None))(
        input_data,
        initial_weights,
        activation_function,
        volterra_coefficients)


def generate_trajectory(
        input_sequence,
        initial_weights,
        activation_function,
        volterra_coefficients):
    """
    generate a single trajectory given an input sequence, initial weights
    and the "meta" plasticity coefficients
    """

    def step(weights, inputs):
        return network_step(
            inputs,
            weights,
            activation_function,
            volterra_coefficients)

    final_weights, activity_trajec = jax.lax.scan(
        step, initial_weights, input_sequence
        )
    return activity_trajec


def network_step(
        inputs,
        weights,
        activation_function,
        volterra_coefficients):

    outputs = activation_function(inputs @ weights)

    m, n = weights.shape
    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(outputs, (n,)), indexing="ij"
    )

    dw = vmap(synapse_step, in_axes=(0, 0, 0, None))(
            reshape(in_grid, (m * n, 1)),
            reshape(out_grid, (m * n, 1)),
            reshape(weights, (m * n, 1)),
            volterra_coefficients)

    dw = reshape(dw, (m, n))
    assert (
        dw.shape == weights.shape
    ), \
        "dw and w should be of the same shape to prevent broadcasting while " \
        "adding"
    weights += dw

    return (weights, outputs)


def synapse_step(pre, post, weight, volterra_coefficients):
    synapse_tensor = get_synapse_tensor(pre, post, weight)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw


def get_synapse_tensor(pre, post, weight):
    synapse_tensor = jnp.outer(
        jnp.outer(
            jnp.array([pre**0, pre**1, pre**2]),
            jnp.array([post**0, post**1, post**2])),
        jnp.array([weight**0, weight**1, weight**2]))

    synapse_tensor = jnp.reshape(synapse_tensor, (3, 3, 3))
    return synapse_tensor
