import jax
import jax.numpy as jnp
from jax.lax import reshape
from jax import vmap


def generate_trajectories(
    input_data,
    initial_weights,
    connectivity_matrix,
    plasticity_parameters,
    plasticity_function,
):

    return vmap(generate_trajectory, in_axes=(0, None, None, None, None))(
        input_data,
        initial_weights,
        connectivity_matrix,
        plasticity_parameters,
        plasticity_function,
    )


def generate_trajectory(
    input_sequence,
    initial_weights,
    connectivity_matrix,
    plasticity_parameters,
    plasticity_function,
):
    """
    generate a single activity trajectory given an input sequence, initial weights
    and the "meta" plasticity coefficients
    """

    initial_weights = jnp.multiply(initial_weights, connectivity_matrix)

    def step(weights, inputs):
        return network_step(
            inputs,
            weights,
            connectivity_matrix,
            plasticity_parameters,
            plasticity_function,
        )

    final_weights, activity_trajec = jax.lax.scan(step, initial_weights, input_sequence)
    return activity_trajec, final_weights


def network_step(
    inputs, weights, connectivity_matrix, plasticity_parameters, plasticity_function
):

    outputs = jax.nn.sigmoid(inputs @ weights)

    m, n = weights.shape
    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(outputs, (n,)), indexing="ij"
    )

    vfun = vmap(plasticity_function, in_axes=(0, 0, 0, None))

    dw = vmap(vfun, in_axes=(1, 1, 1, None), out_axes=1)(
        in_grid, out_grid, weights, plasticity_parameters
    )
    dw = jnp.multiply(dw, connectivity_matrix)

    assert dw.shape == weights.shape, (
        "dw and w should be of the same shape to prevent broadcasting while " "adding"
    )
    weights += dw

    return (weights, outputs)
