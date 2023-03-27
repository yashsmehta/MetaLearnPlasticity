import jax
import jax.numpy as jnp
from jax.lax import reshape
from jax import vmap

"""
the only change from network.py is just in the network_step function, where we are using the dan activity to calculate the dw
in place of the usual y activity as done originally in single layer networks.
"""
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
    and the "meta" plasticity coefficients. sparse connectivity_matrix matrix is applied
    to the weights in the first layer (kc --> mbon)
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

    final_weights, (dw_trajectory, weight_trajectory, activity_trajectory) = jax.lax.scan(step, initial_weights, input_sequence)
    return (dw_trajectory, weight_trajectory, activity_trajectory), final_weights


def network_step(
    inputs, weights, connectivity_matrix, plasticity_parameters, plasticity_function
):
    m, n = weights.shape
    key = jax.random.PRNGKey(0)
    mbon_dan_weights = 0.1 * jnp.multiply(jax.random.normal(key, (n,)), jnp.eye(n))

    mbon_activity = jax.nn.sigmoid(inputs @ weights)
    dan_activity = jax.nn.sigmoid(mbon_activity @ mbon_dan_weights)
    # dan_activity = jax.nn.tanh(mbon_activity @ mbon_dan_weights)

    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(dan_activity, (n,)), indexing="ij"
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

    return (weights, (dw, weights, dan_activity))

