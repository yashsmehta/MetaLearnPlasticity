
import jax.numpy as jnp
import jax
from jax import vmap
from plasticity import losses, synapse, utils, inputs, network
import unittest


class TestTrainingLoop(unittest.TestCase):

    teacher_coefficients, teacher_plasticity_function = synapse.init_volterra("oja")
    student_coefficients, student_plasticity_function = synapse.init_volterra("oja")

    key = jax.random.PRNGKey(42)

    connectivity_matrix = utils.generate_random_connectivity(
        key, 5, 5, sparsity=1
    )
    winit_teacher = utils.generate_gaussian(
        key, (5,5), scale=0.1
    )

    winit_student = utils.generate_gaussian(
        key, (5,5), scale=0.1
    )

    mus, sigmas = inputs.generate_input_parameters(
        key, 5, 10, firing_fraction=0.5
    )

    odors_tensor = jax.random.choice(
        key, jnp.arange(10), shape=(1, 50)
    )
    keys_tensor = jax.random.split(key, num=(1 * 50))
    keys_tensor = keys_tensor.reshape(1, 50, 2)

    input_data = inputs.generate_sparse_inputs(mus, sigmas, odors_tensor, keys_tensor)

    teacher_trajectories = network.generate_trajectories(
        input_data,
        winit_teacher,
        connectivity_matrix,
        teacher_coefficients,
        teacher_plasticity_function,
    )
    student_trajectories = network.generate_trajectories(
        input_data,
        winit_student,
        connectivity_matrix,
        student_coefficients,
        student_plasticity_function,
    )

    loss_value_and_grad = jax.value_and_grad(
        losses.mse_plasticity_coefficients, argnums=2
    )

    loss, meta_grads = loss_value_and_grad(
        input_data[0],
        teacher_trajectories[0],
        student_coefficients,
        student_plasticity_function,
        winit_student,
        connectivity_matrix,
    )

    print("loss", loss)
    print("teacher_trajectory start", teacher_trajectories[0][0])
    print("teacher_trajectory finish", teacher_trajectories[0][-1])

    def test_zero_gradients(self):
        self.assertEqual(self.loss, 0)
        self.assertFalse(jnp.any(self.meta_grads))

    def test_activity_trajectory(self):
        self.assertEqual(self.teacher_trajectories.shape, self.student_trajectories.shape)
        trajectories_diff = self.teacher_trajectories - self.student_trajectories 
        self.assertFalse(jnp.any(trajectories_diff))

if __name__ == "__main__":
    unittest.main()