import jax
import jax.numpy as jnp
from functools import partial
import optax
from plasticity import network


def compute_mse(student_trajectory, teacher_trajectory):
    """
    takes a single student and teacher trajectory and return the MSE loss
    between them
    """
    return jnp.mean(optax.l2_loss(student_trajectory, teacher_trajectory))


@partial(jax.jit, static_argnames=["student_plasticity_function"])
def mse_plasticity_coefficients(
    keys_tensor,
    input_sequence,
    reward_probabilities,
    teacher_trajectory,
    student_coefficients,
    student_plasticity_function,
    winit_student,
    connectivity_matrix,
):
    """
    generates the student trajectory using corresponding coefficients and then
    calls function to compute loss to the given teacher trajectory
    """

    (_, _, student_trajectory), _ = network.generate_trajectory(
        keys_tensor,
        input_sequence,
        reward_probabilities,
        student_coefficients,
        student_plasticity_function,
        winit_student,
        connectivity_matrix,
    )
    loss = compute_mse(student_trajectory, teacher_trajectory)
    # add a L1 regularization term to the loss
    # loss += 5e-6 * jnp.sum(jnp.abs(student_coefficients))
    return loss
