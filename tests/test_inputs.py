import jax.numpy as jnp
import jax
from plasticity import inputs
import unittest


class TestInputs(unittest.TestCase):

    def test_inputs(self):

        mus = jnp.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        sigmas = jnp.array([
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ],
            [
                [0.1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.1]
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0.001]
            ]
        ])

        random_key = jax.random.PRNGKey(42)
        for i in range(10):
            random_key, _ = jax.random.split(random_key)
            x = inputs.sample_inputs(mus, sigmas, i % 3, random_key)
            print(f"k={i%3}, x={x}")