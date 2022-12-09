import os
import jax
import jax.numpy as jnp
from jax import jit
import optax
import time
import numpy as np

import network

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def compute_loss(student_trajectory, teacher_trajectory):
    return jnp.mean(optax.l2_loss(student_trajectory, teacher_trajectory))


@jit
def compute_plasticity_coefficients_loss(
        input_sequence,
        initial_weights,
        oja_coefficients,
        student_coefficients):
    """
    function will generate the teacher trajectory and student trajectory
    using corresponding coefficients and then compute the mse loss between them
    """
    teacher_trajectory = network.generate_trajectory(
        input_sequence, initial_weights, oja_coefficients
    )

    student_trajectory = network.generate_trajectory(
        input_sequence, initial_weights, student_coefficients
    )
    loss = compute_loss(student_trajectory, teacher_trajectory)
    return loss


if __name__ == "__main__":
    num_trajec, len_trajec = 200, 100
    input_dim, output_dim = 100, 10
    epochs = 50

    key = jax.random.PRNGKey(0)
    initial_weights = generate_gaussian(key,
        (input_dim, output_dim),
        scale=1 / (input_dim + output_dim))

    key, key2 = jax.random.split(key)
    # (num_trajectories, length_trajectory, input_dim)
    input_data = generate_gaussian(
        key,
        (num_trajec, len_trajec, input_dim),
        scale=0.1)

    oja_coefficients = np.zeros((3, 3, 3))
    oja_coefficients[1][1][0] = 1
    oja_coefficients[0][2][1] = -1
    # student_coefficients = generate_gaussian(key2, (3, 3, 3), scale=1e-4)
    student_coefficients = jnp.zeros((3,3,3))

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_coefficients)

    for _ in range(epochs):
        loss_i = 0
        for j in range(num_trajec):
            start = time.time()
            input_sequence = input_data[j]

            loss, grads = jax.value_and_grad(compute_plasticity_coefficients_loss, argnums=3)(
                input_sequence,
                initial_weights,
                oja_coefficients,
                student_coefficients)
            loss_i += loss

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                student_coefficients)

            student_coefficients = optax.apply_updates(
                student_coefficients,
                updates)

        print("loss ", loss_i)

    print("oja coefficients\n", oja_coefficients)
    print("student coefficients\n", student_coefficients)