import plasticity.inputs as inputs
from plasticity import utils, synapse
import plasticity.behavior.network as network
import plasticity.behavior.losses as losses
import jax
import optax
from tqdm import tqdm
from jax.random import split
import time


if __name__ == "__main__":
    epochs = 5
    num_blocks, trials_per_block = 2, 5
    reward_ratios = ((0.1, 0.9), (0.9, 0.1))
    assert len(reward_ratios) == num_blocks, print("length of reward ratios should be equal to number of blocks!")

    input_dim, output_dim = 3, 1
    key = jax.random.PRNGKey(2)

    fly_plasticity_coeff, plasticity_func = synapse.init_reward_volterra(init="reward")
    insilico_plasticity_coeff, _ = synapse.init_reward_volterra(key, init="random")

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

    xs, fly_ys, rewards, exp_rewards = network.simulate_fly_experiment(
        key, winit, fly_plasticity_coeff, plasticity_func, mus, sigmas, reward_ratios, trials_per_block
    )
   
    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(insilico_plasticity_coeff)

    for epoch in range(epochs):
        loss = 0
        start = time.time()
        print("Epoch {}:".format(epoch + 1))
        loss, meta_grads = loss_value_and_grad(winit, insilico_plasticity_coeff, plasticity_func, xs, rewards, exp_rewards, fly_ys)

        updates, opt_state = optimizer.update(
            meta_grads, opt_state, insilico_plasticity_coeff
        )

        insilico_plasticity_coeff = optax.apply_updates(insilico_plasticity_coeff, updates)
