
import jax.numpy as jnp
import jax
import numpy as np
import sklearn.metrics
from plasticity import inputs 
import network

def get_r2_score(
    winit,
    connectivity_matrix,
    student_coefficients,
    student_plasticity_function,
    teacher_coefficients,
    teacher_plasticity_function,
):
    """
    returns the r2 scores calculated on the weight and activity trajectories
    for now, only coded for num_odors = 2
    """
    key = jax.random.PRNGKey(66)
    num_trajec_test, len_trajec = 10, 60
    reward_ratios_seq = ((0.5, 0.5), (0.2, 0.8), (0.4, 0.6))
    num_odors = 2
    input_dim = connectivity_matrix.shape[0]
    mus, sigmas = inputs.generate_input_parameters(
        key, input_dim, num_odors, firing_fraction=0.1
    )
    key, key2 = jax.random.split(key)
    odors_tensor = jax.random.choice(
        key2, jnp.arange(num_odors), shape=(num_trajec_test, len_trajec)
    )
    keys_tensor = jax.random.split(key, num=(num_trajec_test * len_trajec))
    keys_tensor = keys_tensor.reshape(num_trajec_test, len_trajec, 2)
    input_data = inputs.generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor)
    reward_prob_tensor = inputs.get_reward_prob_tensor(odors_tensor, reward_ratios_seq)
    # note: for calculating r2 score, we want to calculate it with the weight
    # trajectories and not activity trajectories
    (
        _,
        weight_teacher_trajectories,
        activity_teacher_trajectories,
    ), _ = network.generate_trajectories(
        keys_tensor,
        input_data,
        reward_prob_tensor,
        teacher_coefficients,
        teacher_plasticity_function,
        winit,
        connectivity_matrix,
    )

    (
        _,
        weight_student_trajectories,
        activity_student_trajectories,
    ), _ = network.generate_trajectories(
        keys_tensor,
        input_data,
        reward_prob_tensor,
        student_coefficients,
        student_plasticity_function,
        winit,
        connectivity_matrix,
    )
    r2_score_weight, r2_score_activity = np.zeros(num_trajec_test), np.zeros(
        num_trajec_test
    )

    for i in range(num_trajec_test):
        r2_score_weight[i] = sklearn.metrics.r2_score(
            weight_teacher_trajectories[i].reshape(len_trajec, -1),
            weight_student_trajectories[i].reshape(len_trajec, -1),
        )
        r2_score_activity[i] = sklearn.metrics.r2_score(
            activity_teacher_trajectories[i], activity_student_trajectories[i]
        )
    return jnp.mean(r2_score_activity), jnp.mean(r2_score_weight)

