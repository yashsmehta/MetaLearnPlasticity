from functools import partial
import jax
import jax.numpy as jnp
import network
import numpy as np
import optax
import synapse
from tqdm import tqdm
import time
from utils import generate_gaussian


def compute_loss(student_trajectory, teacher_trajectory):
    """
    takes a single student and teacher trajectory and return the MSE loss
    between them
    """
    return jnp.mean(optax.l2_loss(student_trajectory, teacher_trajectory))


@partial(jax.jit, static_argnames=['student_plasticity_function'])
def compute_plasticity_coefficients_loss(
        input_sequence,
        teacher_trajectory,
        student_coefficients,
        student_plasticity_function,
        winit_student):
    """
    generates the student trajectory using corresponding coefficients and then
    calls function to compute loss to the given teacher trajectory
    """

    student_trajectory = network.generate_trajectory(
        input_sequence,
        winit_student,
        student_coefficients,
        student_plasticity_function)

    loss = compute_loss(student_trajectory, teacher_trajectory)

    return loss


if __name__ == "__main__":
    num_trajec, len_trajec = 50, 500 
    input_dim, output_dim = 200, 200 
    epochs = 2

    # step size of the gradient descent on the initial student weights
    winit_step_size = 0.1

    teacher_coefficients, teacher_plasticity_function = \
        synapse.init_volterra('oja')

    key = jax.random.PRNGKey(0)
    student_coefficients, student_plasticity_function = \
        synapse.init_volterra('random', key)

    key, key2 = jax.random.split(key)

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
        argnums=(2, 4))

    # precompute all teacher trajectories
    start = time.time()
    teacher_trajectories = network.generate_trajectories(
        input_data,
        winit_teacher,
        teacher_coefficients,
        teacher_plasticity_function)

    print("teacher trajecties generated in: {}s ".format(
        round(time.time() - start, 3)))

    for epoch in range(epochs):
        loss = 0
        start = time.time()
        diff_w.append(np.absolute(winit_teacher - winit_student))
        print("Epoch {}:".format(epoch+1))
        for j in tqdm(range(num_trajec), "#trajectory"):

            input_sequence = input_data[j]
            teacher_trajectory = teacher_trajectories[j]

            loss_j, (meta_grads, grads_winit) = loss_value_grad(
                input_sequence,
                teacher_trajectory,
                student_coefficients,
                student_plasticity_function,
                winit_student)

            loss += loss_j
            winit_student -= winit_step_size * grads_winit
            updates, opt_state = optimizer.update(
                meta_grads,
                opt_state,
                student_coefficients)

            student_coefficients = optax.apply_updates(
                student_coefficients,
                updates)

        print("Epoch Time: {}s".format(round((time.time() - start), 3)))
        print("average loss per trajectory: ", round((loss / num_trajec), 10))
        print()

    # np.savez("expdata/winit/sameinit", diff_w)
    print("teacher coefficients\n", teacher_coefficients)
    print("student coefficients\n", student_coefficients)
