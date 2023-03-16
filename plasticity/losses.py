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
    input_sequence,
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
        input_sequence,
        winit_student,
        connectivity_matrix,
        student_coefficients,
        student_plasticity_function,
    )
    loss = compute_mse(student_trajectory, teacher_trajectory)
    return loss
