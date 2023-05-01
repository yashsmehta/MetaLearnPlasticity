import jax
import jax.numpy as jnp
import plasticity.inputs as inputs
from plasticity import utils, synapse
import plasticity.behavior.network as network
import optax
from tqdm import tqdm
import numpy as np
from jax.random import split
import time


if __name__ == "__main__":
    num_blocks, trials_per_block = 2, 5
    reward_ratios = ((0.1, 0.9), (0.9, 0.1))
    assert len(reward_ratios) == num_blocks, print("length of reward ratios should be equal to number of blocks!")

    input_dim, output_dim = 3, 1
    key = jax.random.PRNGKey(2)

    plasticity_coeff, plasticity_func = synapse.init_reward_volterra("reward")

    key, key2 = split(key)

    winit = utils.generate_gaussian(
        key, (input_dim, output_dim), scale=1 / (input_dim + output_dim)
    )
    key, key2 = split(key)

    num_odors = 2
    mus, sigmas = inputs.generate_input_parameters(
        key, input_dim, num_odors, firing_fraction=0.2
    )

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()

    xs, sampled_ys, rewards, exp_rewards = network.simulate_fly_experiment(
        key, winit, plasticity_coeff, plasticity_func, mus, sigmas, reward_ratios, trials_per_block
    )

    ys = network.simulate_insilico_experiment(key, winit, plasticity_coeff, plasticity_func, xs, rewards, exp_rewards)


    print("x for (block1) \n", (xs[0]) )
    print()
    print("choices for (block1) \n", (sampled_ys[0]))
    print()
    print("rewards for (block1) \n", (rewards[0]))