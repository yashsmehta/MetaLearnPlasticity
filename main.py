from functools import partial
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

import network


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def compute_loss(student_trajectory, teacher_trajectory):
    """
    takes a single student and teacher trajectory and return the MSE loss
    between them
    """
    return jnp.mean(optax.l2_loss(student_trajectory, teacher_trajectory))


@partial(jax.jit, static_argnames=['activation_function'])
def compute_plasticity_coefficients_loss(
        input_sequence,
        teacher_trajectory,
        student_coefficients,
        winit_student,
        activation_function):
    """
    generates the student trajectory using corresponding coefficients and then
    calls function to compute loss to the given teacher trajectory
    """

    student_trajectory = network.generate_trajectory(
        input_sequence,
        winit_student,
        activation_function,
        student_coefficients)

    loss = compute_loss(student_trajectory, teacher_trajectory)

    return loss


if __name__ == "__main__":
    num_trajec, len_trajec = 50, 50
    input_dim, output_dim = 100, 100
    epochs = 30

    # step size of the gradient descent on the initial student weights
    winit_step_size = 0.1

    activation_function = jax.nn.sigmoid

    key = jax.random.PRNGKey(0)
    winit_teacher = generate_gaussian(
                        key,
                        (input_dim, output_dim),
                        scale=1 / (input_dim + output_dim))

    winit_student = generate_gaussian(
                        key,
                        (input_dim, output_dim),
                        scale=1 / (input_dim + output_dim))
    key, key2 = jax.random.split(key)

    # (num_trajectories, length_trajectory, input_dim)
    input_data = generate_gaussian(
        key,
        (num_trajec, len_trajec, input_dim),
        scale=0.1)

    key, key2 = jax.random.split(key)

    # use Oja's rule for teacher coefficients
    teacher_coefficients = np.zeros((3, 3, 3))
    teacher_coefficients[1][1][0] = 1
    teacher_coefficients[0][2][1] = -1

    # initialize student coefficients randomly
    student_coefficients = generate_gaussian(key, (3, 3, 3), scale=1e-5)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_coefficients)

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()
    diff_w = []

    loss_value_grad = jax.value_and_grad(
        compute_plasticity_coefficients_loss,
        argnums=(2, 3))

    # precompute all teacher trajectories
    teacher_trajectories = network.generate_trajectories(
        input_data,
        winit_teacher,
        activation_function,
        teacher_coefficients)

    for epoch in tqdm(range(epochs), "epoch"):
        loss = 0
        start = time.time()
        diff_w.append(np.absolute(winit_teacher - winit_student))
        for j in tqdm(range(num_trajec), "trajectory"):

            input_sequence = input_data[j]
            teacher_trajectory = teacher_trajectories[j]

            loss_j, (meta_grads, grads_winit) = loss_value_grad(
                input_sequence,
                teacher_trajectory,
                student_coefficients,
                winit_student,
                activation_function)

            loss += loss_j
            winit_student -= winit_step_size * grads_winit
            updates, opt_state = optimizer.update(
                meta_grads,
                opt_state,
                student_coefficients)

            student_coefficients = optax.apply_updates(
                student_coefficients,
                updates)

        print("epoch {} in {}s".format(epoch, round((time.time() - start), 3)))
        print("average loss per trajectory: ", round((loss / num_trajec), 10))
        print()

    np.savez("expdata/winit/sameinit", diff_w)
    print("teacher coefficients\n", teacher_coefficients)
    print("student coefficients\n", student_coefficients)
