import jax
import jax.numpy as jnp
from jax.lax import reshape
from jax import vmap


def generate_trajectories(
        input_data,
        initial_weights,
        plasticity_parameters,
        plasticity_function,
        activation_function):

    return vmap(generate_trajectory, in_axes=(0, None, None, None, None))(
        input_data,
        initial_weights,
        plasticity_parameters,
        plasticity_function,
        activation_function)


def generate_trajectory(
        input_sequence,
        initial_weights,
        plasticity_parameters,
        plasticity_function,
        activation_function):
    """
    generate a single trajectory given an input sequence, initial weights
    and the "meta" plasticity coefficients
    """

    def step(weights, inputs):
        return network_step(
            inputs,
            weights,
            plasticity_parameters,
            plasticity_function,
            activation_function)

    final_weights, activity_trajec = jax.lax.scan(
        step, initial_weights, input_sequence
        )
    return activity_trajec


def network_step(
        inputs,
        weights,
        plasticity_parameters,
        plasticity_function,
        activation_function):

    outputs = activation_function(inputs @ weights)

    m, n = weights.shape
    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(outputs, (n,)), indexing="ij"
    )

    dw = vmap(plasticity_function, in_axes=(0, 0, 0, None))(
            reshape(in_grid, (m * n, 1)),
            reshape(out_grid, (m * n, 1)),
            reshape(weights, (m * n, 1)),
            plasticity_parameters)

    dw = reshape(dw, (m, n))
    assert (
        dw.shape == weights.shape
    ), \
        "dw and w should be of the same shape to prevent broadcasting while " \
        "adding"
    weights += dw

    return (weights, outputs)
