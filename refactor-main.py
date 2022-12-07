import os
import jax
import jax.numpy as jnp
from jax import jit, vmap
import optax
import time
import numpy as np
import time
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_gaussian(key, shape, scale=0.1):
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


class Network:
    def __init__(self, non_linear=True, jit=True):
        self.non_linear = non_linear
        if jit:
            self.forward = jax.jit(self.forward)
            self.generate_trajectory = jax.jit(self.generate_trajectory)

    def generate_trajectory(
        self, input_sequence, initial_weights, volterra_coefficients
    ):
        def step(weights, inputs):
            return self.update_weights(inputs, weights, volterra_coefficients)

        final_weights, activities = jax.lax.scan(step, initial_weights, input_sequence)
        return activities

    def update_weights(self, inputs, weights, volterra_coefficients):
        activation = self.forward(inputs, weights)
        # inefficient update
        dw = 0
        for index in range(len(volterra_coefficients)):
            i, j, k = utils.A_index_to_powers(index)
            dw += volterra_coefficients[index] * jnp.multiply(
                jnp.outer(inputs**i, activation**j), weights**k
            )

        if dw.shape != weights.shape:
            raise Exception(
                "dw and w should be of the same shape to prevent broadcasting while adding"
            )
        weights += dw
        # weights = 0.99 * weights
        return (weights, activation)

    def forward(self, inputs, weights):
        activatation = inputs @ weights
        if self.non_linear:
            activatation = jax.nn.sigmoid(activatation)
        return activatation


def compute_loss(student_trajectory, teacher_trajectory):
    return jnp.mean(optax.l2_loss(student_trajectory, teacher_trajectory))


def compute_plasticity_coefficients_loss(
    input_sequence, initial_weights, oja_coefficients, student_coefficients
):
    network = Network()
    teacher_trajectory = network.generate_trajectory(
        input_sequence, initial_weights, oja_coefficients
    )

    student_trajectory = network.generate_trajectory(
        input_sequence, initial_weights, student_coefficients
    )
    loss = compute_loss(student_trajectory, teacher_trajectory)
    return loss


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    num_trajec, len_trajec = 10, 500
    epochs = 2

    initial_weights = generate_gaussian(key, (100, 10))  # (input_dim, output_dim)
    input_data = generate_gaussian(
        key, (num_trajec, len_trajec, 100)
    )  # (num_trajectories, length_trajectory, input_dim)

    oja_coefficients = np.zeros((27,))
    oja_coefficients[utils.powers_to_A_index(1, 1, 0)] = 1
    oja_coefficients[utils.powers_to_A_index(0, 2, 1)] = -1
    student_coefficients = generate_gaussian(key, (27,), scale=1e-4)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(student_coefficients)

    for _ in range(epochs):
        for j in range(num_trajec):
            input_sequence = input_data[j]

            loss, grads = jax.value_and_grad(
                compute_plasticity_coefficients_loss, argnums=3
            )(input_sequence, initial_weights, oja_coefficients, student_coefficients)
            print("loss ", loss)
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                student_coefficients,
            )
            student_coefficients = optax.apply_updates(student_coefficients, updates)
