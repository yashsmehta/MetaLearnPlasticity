from inputs import sample_inputs
import jax.numpy as jnp
import jax


if __name__ == '__main__':

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
        x = sample_inputs(mus, sigmas, i%3, random_key)
        print(f"k={i%3}, x={x}")
