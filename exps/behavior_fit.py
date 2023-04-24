import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import time
import numpy as np
from plasticity import inputs, utils, synapse, network, losses


def train_epoch(params, opt_state):
    total_loss = 0
    for j in tqdm(range(num_trajec), "#trajectory"):
        input_sequence, reward_probabilities, teacher_trajectory, keys = (
            input_data[j],
            reward_prob_tensor[j],
            teacher_trajectories[j],
            keys_tensor[j],
        )

        loss_j, params_grads = loss_value_and_grad(
            keys,
            input_sequence,
            reward_probabilities,
            teacher_trajectory,
            student_coefficients,
            student_plasticity_function,
            winit,
            connectivity_matrix,
        )

        total_loss += loss_j
        updates, opt_state = optimizer.update(params_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return total_loss, params, opt_state


if __name__ == "__main__":
    input_dim, output_dim = 50, 1
    num_trajec, len_trajec = 10, 60
    reward_ratios_seq = ((0.5, 0.5), (0.1, 0.9), (0.3, 0.7))
    epochs = 10
    key = jax.random.PRNGKey(0)

    teacher_coefficients, teacher_plasticity_function = synapse.init_dopamine_volterra(
        "reward", key
    )
    student_coefficients, student_plasticity_function = synapse.init_dopamine_volterra(
        "random", key
    )

    key, _ = jax.random.split(key)

    connectivity_matrix = utils.generate_random_connectivity(
        key, input_dim, output_dim, sparsity=1
    )
    key, key2 = jax.random.split(key)

    winit_true = utils.generate_gaussian(
        key, (input_dim, output_dim), scale=10 / input_dim
    )
    winit = utils.generate_gaussian(key, (input_dim, output_dim), scale=10 / input_dim)

    key, key2 = jax.random.split(key)

    start = time.time()
    num_odors = 2

    mus, sigmas = inputs.generate_input_parameters(
        key, input_dim, num_odors, firing_fraction=0.4
    )

    odors_tensor = jax.random.choice(
        key2, jnp.arange(num_odors), shape=(num_trajec, len_trajec)
    )
    # reward_ratio_seq defines the reward ratio of odor A to odor B for each block of trials
    reward_prob_tensor = inputs.get_reward_prob_tensor(odors_tensor, reward_ratios_seq)

    keys_tensor = jax.random.split(key, num=(num_trajec * len_trajec))
    keys_tensor = keys_tensor.reshape(num_trajec, len_trajec, 2)

    input_data = inputs.generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor)

    (
        dw_trajectories,
        weight_trajectories,
        teacher_trajectories,
    ), final_weights = network.generate_trajectories(
        keys_tensor,
        input_data,
        reward_prob_tensor,
        teacher_coefficients,
        teacher_plasticity_function,
        winit_true,
        connectivity_matrix,
    )

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_coefficients)

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()
    diff_w = []

    loss_value_and_grad = jax.value_and_grad(
        losses.mse_plasticity_coefficients, argnums=4
    )

    expdata = {
        "loss": np.zeros(epochs),
        "r2_score": np.zeros(epochs),
        "epoch": np.arange(epochs),
    }

    print(
        "initial r2 score:",
        utils.get_r2_score(
            winit_true,
            connectivity_matrix,
            student_coefficients,
            student_plasticity_function,
            teacher_coefficients,
            teacher_plasticity_function,
        ),
    )


logger = []

for epoch in range(epochs):
    start = time.time()
    print(f"epoch {epoch + 1}:")

    loss, student_coefficients, opt_state = train_epoch(student_coefficients, opt_state)
    expdata["loss"][epoch] = loss
    logger.append(student_coefficients)

    expdata["r2_score"][epoch], r2_weight = utils.get_r2_score(
        winit,
        connectivity_matrix,
        student_coefficients,
        student_plasticity_function,
        teacher_coefficients,
        teacher_plasticity_function,
    )

    print(f"epoch time: {round((time.time() - start), 1)}s")
    print(f"average LOSS per trajectory: {loss / num_trajec}")
    print(f"Activities R2 score: {expdata['r2_score'][epoch]}")
    print(f"Weights R2 score: {r2_weight}\n")
