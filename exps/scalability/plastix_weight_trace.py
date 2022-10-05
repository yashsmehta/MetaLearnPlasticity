import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
import time

import plastix as px

# import exps.dataloaders.familiarity_detection_ds as ds


def generate_gaussian(key, shape, scale=0.1):
    """
    generate gaussian data, called for input data, x of particular length that will be used to simulate
    the trajectory
    random initialization of the student and teacher network edge weights
    """
    return scale * jax.random.normal(key, (shape))


def generate_weight_trajectory(layer, state, parameters, x):
    """
    simulate the network for t time steps with corresponding edge & node kernel parameters.
    shape(x): (trajectory_length, input_dim)
    return list of weight jnp arrays, i.e. weight trajectory for this dataset(x)
    """
    weight_trajectory = []

    for i in range(len(x)):
        state.input_nodes.rate = x[i]
        state = layer.update_state(state, parameters, use_jit=True)
        parameters = layer.update_parameters(state, parameters, use_jit=True)
        weight_trajectory.append(parameters.edges.weight)

    return weight_trajectory


def trajectory_loss(layer, state, parameters, x, weight_trajectory):
    """
    function that simulates the student network with the same sequence of data
    and calculates the loss with weight trajectory
    """
    mse_loss = 0
    parameters.edges.weight = jax.lax.stop_gradient(parameters.edges.weight)

    for i in range(len(x)):
        state.input_nodes.rate = x[i]
        state = layer.update_state(state, parameters)
        parameters = layer.update_parameters(state, parameters)
        mse_loss += jnp.mean(
            optax.l2_loss(parameters.edges.weight, weight_trajectory[i])
        )

    return mse_loss


def create_network_layer(m, n):
    """
    function to init  plastix layer (our network)
    will be called twice, for student and teacher
    note: these are just dummy initializations that will be over-ridden
    by an explicit init outside in the main function.
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
    num_trajectories = 5
    length = 50 
    m, n = 5, 1
    teacher_layer, teacher_state, teacher_parameters = create_network_layer(
        m, n
    )
    # get corresponding ground truth plasticity rule (Oja's) for the teacher network
    A = np.zeros((3, 3, 3))
    A[1][1][0] = 1
    A[0][2][1] = -1
    teacher_parameters.edges.coefficient_matrix = A
    # print("A Oja's:", teacher_parameters.edges.coefficient_matrix)

    student_layer, student_state, student_parameters = create_network_layer(
        m, n
    )
    # student_parameters.edges.coefficient_matrix = 1 * jax.random.normal(
    #     key, (3, 3, 3)
    # )
    student_parameters.edges.coefficient_matrix = jnp.zeros((3,3,3))

    print("A init:", student_parameters.edges.coefficient_matrix[1][1][0])
    print(student_parameters.edges.coefficient_matrix[0][2][1])
    ainit = student_parameters.edges.coefficient_matrix
    key, _ = jax.random.split(key)

    # same random initialization of the weights at the edges for student and teacher network
    teacher_parameters.edges.weight = generate_gaussian(
        key, (m, n, 1), scale=1 / (m + n)
    )
    student_parameters.edges.weight = generate_gaussian(
        key, (m, n, 1), scale=1 / (m + n)
    )

    loss = Partial((trajectory_loss), student_layer)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_parameters.edges.coefficient_matrix)

    for epoch in range(meta_epochs):
        key = jax.random.PRNGKey(0)
        for _ in range(num_trajectories):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (length, m), scale=0.1)
            # JIT this!
            weight_trajectory = generate_weight_trajectory(
                teacher_layer, teacher_state, teacher_parameters, x
            )
            # print("weight trajectory", weight_trajectory)
            loss_t, grads = jax.value_and_grad(loss, argnums=1)(
                student_state, student_parameters, x, weight_trajectory
            )
            print("loss: ", loss_t)
            print("grads coefficient matrix: ", grads.edges.coefficient_matrix)
            updates, opt_state = optimizer.update(
                grads.edges.coefficient_matrix,
                opt_state,
                student_parameters.edges.coefficient_matrix,
            )
            student_parameters.edges.coefficient_matrix = optax.apply_updates(
                student_parameters.edges.coefficient_matrix, updates
            )

        # print("finished epoch: {}".format(epoch))

    print("A final:", student_parameters.edges.coefficient_matrix)
    print("A difference:", student_parameters.edges.coefficient_matrix - ainit)
    mat = student_parameters.edges.coefficient_matrix - ainit
    # print(mat[1][1][0])
    # print(mat[0][2][1])
    # print("edge parameters:", parameters.edges.weight)
    # print("edge state:", state.edges.signal)
    # print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
