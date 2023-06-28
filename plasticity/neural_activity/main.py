import jax
import jax.numpy as jnp
from plasticity import inputs, utils, synapse, network, losses
import optax
from tqdm import tqdm
import time


if __name__ == "__main__":
    num_trajec, len_trajec = 200, 100
    # implement a read connectivity function; get the dims and connectivity
    input_dim, output_dim = 10, 10
    key = jax.random.PRNGKey(0)
    epochs = 5

    teacher_coefficients, teacher_plasticity_function = synapse.init_volterra("oja")

    student_coefficients, student_plasticity_function = synapse.init_volterra(
        "random", key
    )

    key, key2 = jax.random.split(key)

    connectivity_matrix = utils.generate_random_connectivity(
        key2, input_dim, output_dim, sparsity=1
    )
    winit_teacher = utils.generate_gaussian(
        key, (input_dim, output_dim), scale=1 / (input_dim + output_dim)
    )

    winit_student = utils.generate_gaussian(
        key, (input_dim, output_dim), scale=1 / (input_dim + output_dim)
    )
    key, key2 = jax.random.split(key)

    start = time.time()
    num_odors = 20
    mus, sigmas = inputs.generate_input_parameters(
        key, input_dim, num_odors, firing_fraction=0.2
    )

    odors_tensor = jax.random.choice(
        key2, jnp.arange(num_odors), shape=(num_trajec, len_trajec)
    )
    keys_tensor = jax.random.split(key, num=(num_trajec * len_trajec))
    keys_tensor = keys_tensor.reshape(num_trajec, len_trajec, 2)

    input_data = inputs.generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_coefficients)

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()
    diff_w = []

    loss_value_and_grad = jax.value_and_grad(
        losses.mse_plasticity_coefficients, argnums=2
    )

    # precompute all teacher trajectories
    start = time.time()
    teacher_trajectories = network.generate_trajectories(
        input_data,
        winit_teacher,
        connectivity_matrix,
        teacher_coefficients,
        teacher_plasticity_function,
    )
    print("teacher trajectories shape", teacher_trajectories.shape)

    for epoch in range(epochs):
        loss = 0
        start = time.time()
        print("Epoch {}:".format(epoch + 1))
        for j in tqdm(range(num_trajec), "#trajectory"):

            input_sequence = input_data[j]
            teacher_trajectory = teacher_trajectories[j]

            loss_j, meta_grads = loss_value_and_grad(
                input_sequence,
                teacher_trajectory,
                student_coefficients,
                student_plasticity_function,
                winit_student,
                connectivity_matrix,
            )

            loss += loss_j
            updates, opt_state = optimizer.update(
                meta_grads, opt_state, student_coefficients
            )

            student_coefficients = optax.apply_updates(student_coefficients, updates)

        print("Epoch Time: {}s".format(round((time.time() - start), 3)))
        print("average loss per trajectory: ", round((loss / num_trajec), 10))
        print()

    # np.savez("expdata/winit/sameinit", diff_w)
    print("teacher coefficients\n", teacher_coefficients)
    print("student coefficients\n", student_coefficients)
