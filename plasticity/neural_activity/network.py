import jax
import jax.numpy as jnp
from jax.lax import reshape
from jax import vmap


def generate_trajectories(
    keys_tensor,
    input_data,
    reward_prob_tensor,
    plasticity_coefficients,
    plasticity_function,
    winit,
    connectivity_matrix,
):

    return vmap(generate_trajectory, in_axes=(0, 0, 0, None, None, None, None))(
        keys_tensor,
        input_data,
        reward_prob_tensor,
        plasticity_coefficients,
        plasticity_function,
        winit,
        connectivity_matrix,
    )


def generate_trajectory(
    keys,
    input_sequence,
    reward_probabilities,
    plasticity_coefficients,
    plasticity_function,
    initial_weights,
    connectivity_matrix,
):
    def step(weights, stimulus):
        key, inputs, reward_probability = stimulus
        return network_step(
            key,
            inputs,
            reward_probability,
            plasticity_coefficients,
            plasticity_function,
            weights,
            connectivity_matrix,
        )

    final_weights, (
        dw_trajectory,
        weight_trajectory,
        activity_trajectory,
    ) = jax.lax.scan(
        step, initial_weights, (keys, input_sequence, reward_probabilities)
    )
    return (dw_trajectory, weight_trajectory, activity_trajectory), final_weights


def network_step(
    key,
    inputs,
    reward_probability,
    plasticity_coefficients,
    plasticity_function,
    weights,
    connectivity_matrix,
):

    outputs = jax.nn.sigmoid(jnp.dot(inputs, weights))

    m, n = weights.shape
    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(outputs, (n,)), indexing="ij"
    )
    reward = jax.random.bernoulli(key, reward_probability)
    decision = jax.random.bernoulli(key, outputs)

    reward_term = (decision * reward - reward_probability)
    # reward_term = decision * reward

    vfun = vmap(plasticity_function, in_axes=(0, None, 0, None))

    dw = vmap(vfun, in_axes=(1, None, 1, None), out_axes=1)(
        in_grid, reward_term, weights, plasticity_coefficients
    )
    dw = jnp.multiply(dw, connectivity_matrix)

    assert (
        dw.shape == weights.shape
    ), "dw and w should be of the same shape to prevent broadcasting while adding"
    weights += dw

    return (weights, (dw, weights, outputs))


"""
lax.scan only returning the activity trajectories
"""
# def generate_trajectories(
#     input_data,
#     initial_weights,
#     connectivity_matrix,
#     plasticity_parameters,
#     plasticity_function,
# ):

#     return vmap(generate_trajectory, in_axes=(0, None, None, None, None))(
#         input_data,
#         initial_weights,
#         connectivity_matrix,
#         plasticity_parameters,
#         plasticity_function,
#     )


# def generate_trajectory(
#     input_sequence,
#     initial_weights,
#     connectivity_matrix,
#     plasticity_parameters,
#     plasticity_function,
# ):
#     """
#     generate a single activity trajectory given an input sequence, initial weights
#     and the "meta" plasticity coefficients
#     """

#     initial_weights = jnp.multiply(initial_weights, connectivity_matrix)

#     def step(weights, inputs):
#         return network_step(
#             inputs,
#             weights,
#             connectivity_matrix,
#             plasticity_parameters,
#             plasticity_function,
#         )

#     final_weights, activity_trajec = jax.lax.scan(step, initial_weights, input_sequence)
#     return activity_trajec, final_weights


# def network_step(
#     inputs, weights, connectivity_matrix, plasticity_parameters, plasticity_function
# ):

#     outputs = jax.nn.sigmoid(inputs @ weights)

#     m, n = weights.shape
#     in_grid, out_grid = jnp.meshgrid(
#         reshape(inputs, (m,)), reshape(outputs, (n,)), indexing="ij"
#     )

#     vfun = vmap(plasticity_function, in_axes=(0, 0, 0, None))

#     dw = vmap(vfun, in_axes=(1, 1, 1, None), out_axes=1)(
#         in_grid, out_grid, weights, plasticity_parameters
#     )
#     dw = jnp.multiply(dw, connectivity_matrix)

#     assert dw.shape == weights.shape, (
#         "dw and w should be of the same shape to prevent broadcasting while " "adding"
#     )
#     weights += dw

#     return (weights, outputs)
