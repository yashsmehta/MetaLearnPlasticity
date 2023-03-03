
import jax.numpy as jnp
import jax
from jax import vmap
from plasticity import losses, synapse, utils, inputs, network
import unittest


class TestZeroGradient(unittest.TestCase):

    def test_zero_gradients(self):

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
            key, 5, 5, firing_fraction=0.1
        )

        odors_tensor = jax.random.choice(
            key, jnp.arange(5), shape=(1, 10)
        )
        keys_tensor = jax.random.split(key, num=(1 * 10))
        keys_tensor = keys_tensor.reshape(1, 10, 2)

        vsample_inputs = vmap(inputs.sample_inputs, in_axes=(None, None, 0, 0))
        vvsample_inputs = vmap(vsample_inputs, in_axes=(None, None, 1, 1))
        input_data = vvsample_inputs(mus, sigmas, odors_tensor, keys_tensor)


        teacher_trajectories = network.generate_trajectories(
            input_data,
            winit_teacher,
            connectivity_matrix,
            teacher_coefficients,
            teacher_plasticity_function,
        )
        loss_value_grad = jax.value_and_grad(
            losses.mse_plasticity_coefficients, argnums=2
        )

        loss, meta_grads = loss_value_grad(
            input_data[0],
            teacher_trajectories[0],
            student_coefficients,
            student_plasticity_function,
            winit_student,
            connectivity_matrix,
        )

        self.assertEqual(loss, 0)
        self.assertFalse(jnp.any(meta_grads))

