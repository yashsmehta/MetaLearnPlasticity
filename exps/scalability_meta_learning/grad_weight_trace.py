import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
import time

import plastix as px

# import exps.dataloaders.familiarity_detection_ds as ds


def generate_gaussian_input(key, length, input_dim, scale=0.01):
    """
    generate input data, x of particular length that will be used to simulate
    the trajectory
    """
    return scale * jax.random.normal(key, (length, input_dim))


def generate_oja_trajectory(layer, state, parameters, x):
    """
    simulate the network for t time steps with Oja's rule
    shape(x): (trajectory_length, input_dim)
    return list of flattened weight jnp arrays (weight trajectory for this dataset(x))
    """
    weight_trajectory = []

    for i in range(len(x)):
        state.input_nodes.rate = x[i]
        state = layer.update_state(state, parameters)
        parameters = layer.update_parameters(state, parameters)
        # weight_trajectory.append(
        #     jnp.array([item for sublist in parameters.edges.weight for item in sublist])
        # )
        weight_trajectory.append(parameters.edges.weight)

    return weight_trajectory


def trajectory_loss(layer, state, parameters, x, weight_trajectory):
    """
    function that simulates the student network with the same sequence of data
    and calculates the loss with weight trajectory
    """
    mse_loss = 0
    for i in range(len(x)):
        state.input_nodes.rate = x[i]
        state = layer.update_state(state, parameters)
        parameters = layer.update_parameters(state, parameters)
        mse_loss += jnp.mean(
            optax.l2_loss(parameters.edges.weight, weight_trajectory[i])
        )

    return mse_loss


def init_network_layer(seed, m, n):
    """
    function to init  plastix layer (our network)
    will be called twice, for student and teacher
    careful: the initialization should be exactly the same for both the networks
    initially: we will need to put a mask on the rest of the meta-parameters in the
    volterra expansion (in RatePolynomialUpdateEdge class)
    """
    layer = px.layers.DenseLayer(
        m,
        n,
        px.kernels.edges.RatePolynomialUpdateEdge(),
        px.kernels.nodes.SumLinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()
    return layer, state, parameters


def main():
    key = jax.random.PRNGKey(0)
    meta_epochs = 1
    num_trajectories = 1
    length = 3

    teacher_layer, teacher_state, teacher_parameters = init_network_layer(
        seed=0, m=5, n=1
    )
    A = np.zeros((3, 3, 3))
    A[1][1][0] = 1
    A[0][2][1] = -1
    teacher_parameters.edges.coefficient_matrix = A
    print("A Oja's:", teacher_parameters.edges.coefficient_matrix)

    student_layer, student_state, student_parameters = init_network_layer(
        seed=0, m=5, n=1
    )
    student_parameters.edges.coefficient_matrix = 0.01 * jax.random.normal(
        key, (3, 3, 3)
    )
    print("A student", student_parameters.edges.coefficient_matrix)

    loss = Partial((trajectory_loss), student_layer)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_parameters.edges.coefficient_matrix)

    for epoch in range(meta_epochs):
        for _ in range(num_trajectories):
            key, _ = jax.random.split(key)
            x = generate_gaussian_input(key, length, input_dim=5)
            weight_trajectory = generate_oja_trajectory(
                teacher_layer, teacher_state, teacher_parameters, x
            )
            print("weight trajectory", weight_trajectory)
            grads = jax.grad(loss, argnums=1)(
                student_state, student_parameters, x, weight_trajectory
            )
            updates, opt_state = optimizer.update(
                grads.edges.coefficient_matrix,
                opt_state,
                student_parameters.edges.coefficient_matrix,
            )
            student_parameters.edges.coefficient_matrix = optax.apply_updates(
                student_parameters.edges.coefficient_matrix, updates
            )

        state = student_layer.update_state(student_state, student_parameters)
        print("finished epoch: {}".format(epoch))
        print("A student", student_parameters.edges.coefficient_matrix)
    # print("edge parameters:", parameters.edges.weight)
    # print("edge state:", state.edges.signal)
    # print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
