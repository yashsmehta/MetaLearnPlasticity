import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from plasticity import inputs, utils, synapse, network, losses


if __name__ == "__main__":
    num_trajec, len_trajec = 2, 100
    input_dim, output_dim = 5, 2 
    key = jax.random.PRNGKey(0)
    epochs = 10

    teacher_coefficients, teacher_plasticity_function = synapse.init_volterra("oja", key)
    student_coefficients, student_plasticity_function = synapse.init_volterra(
        "oja", key
    )

    key, key2 = jax.random.split(key)

    connectivity_matrix = utils.generate_random_connectivity(
        key2, input_dim, output_dim, sparsity=0.5
    )
    winit_true = utils.generate_gaussian(
        key, (input_dim, output_dim), scale=10 / input_dim
    )
    winit = utils.generate_gaussian(
        key, (input_dim, output_dim), scale=1 / input_dim
    )

    key, key2 = jax.random.split(key)

    start = time.time()
    num_odors = 100
    mus, sigmas = inputs.generate_input_parameters(
        key, input_dim, num_odors, firing_fraction=0.1
    )

    odors_tensor = jax.random.choice(
        key2, jnp.arange(num_odors), shape=(num_trajec, len_trajec)
    )
    keys_tensor = jax.random.split(key, num=(num_trajec * len_trajec))
    keys_tensor = keys_tensor.reshape(num_trajec, len_trajec, 2)

    input_data = inputs.generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor)

    optimizer = optax.adam(learning_rate=1e-3)
    params = (student_coefficients, winit)
    opt_state = optimizer.init(params)

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()
    diff_w = []

    loss_value_and_grad = jax.value_and_grad(
        losses.mse_plasticity_coefficients, argnums=(2, 4)
    )

    # precompute all teacher trajectories
    start = time.time()
    (
        dw_trajectories,
        weight_trajectories,
        teacher_trajectories,
    ), final_weights = network.generate_trajectories(
        input_data,
        winit_true,
        connectivity_matrix,
        teacher_coefficients,
        teacher_plasticity_function,
    )

    print("generated teacher trajectories...")

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
        print("epoch {}:".format(epoch + 1))

        for j in tqdm(range(num_trajec), "#trajectory"):

            input_sequence = input_data[j]
            teacher_trajectory = teacher_trajectories[j]

            loss_j, params_grads = loss_value_and_grad(
                input_sequence,
                teacher_trajectory,
                student_coefficients,
                student_plasticity_function,
                winit,
                connectivity_matrix,
            )

            expdata["loss"][epoch] += loss_j
            updates, opt_state = optimizer.update(
                params_grads, opt_state, params
            )

            params = optax.apply_updates(params, updates)

        logger.append(student_coefficients)
        expdata["r2_score"][epoch], r2_weight = utils.get_r2_score(
            winit,
            connectivity_matrix,
            student_coefficients,
            student_plasticity_function,
            teacher_coefficients,
            teacher_plasticity_function,
        )

        print("epoch time: {}s".format(round((time.time() - start), 1)))
        print("average LOSS per trajectory: ", (expdata["loss"][epoch] / num_trajec))
        print("Activities R2 score: ", expdata["r2_score"][epoch])
        print("Weights R2 score: ", r2_weight)
        print()
    
    print("teacher coefficients: ", teacher_coefficients)
    print("student coefficients: ", student_coefficients)

    # np.savez("expdata/sparse_inputs/student_coeffs", np.array(logger))
    # pd.DataFrame(expdata).to_csv("expdata/sparse_inputs/expdf.csv", index=True)
