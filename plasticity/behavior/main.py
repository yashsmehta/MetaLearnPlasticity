import plasticity.inputs as inputs
from plasticity import utils, synapse
import plasticity.behavior.network as network
import plasticity.behavior.losses as losses
from jax.experimental.host_callback import id_print
import jax
import jax.numpy as jnp
from pprint import pprint
import optax
import numpy as np
from jax.random import split
import time


def convert_decisions_to_tensor(nested_list):
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])
    num_trials = num_blocks * trials_per_block

    longest_trial_length = max(
        max(
            [
                [len(nested_list[j][i]) for i in range(trials_per_block)]
                for j in range(num_blocks)
            ]
        )
    )
    tensor = np.full((num_trials, longest_trial_length), np.nan)

    for i in range(num_blocks):
        for j in range(trials_per_block):
            trial = nested_list[i][j]
            for k in range(len(trial)):
                tensor[i * trials_per_block + j][k] = trial[k]

    return tensor


def convert_xs_to_tensor(nested_list):
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])
    num_trials = num_blocks * trials_per_block
    element_dim = 2

    longest_trial_length = max(
        max(
            [
                [len(nested_list[j][i]) for i in range(trials_per_block)]
                for j in range(num_blocks)
            ]
        )
    )

    tensor = np.full((num_trials, longest_trial_length, element_dim), np.nan)

    for i in range(num_blocks):
        for j in range(trials_per_block):
            trial = nested_list[i][j]
            for k in range(len(trial)):
                tensor[i * trials_per_block + j][k] = trial[k]

    return tensor


if __name__ == "__main__":
    num_epochs = 20
    num_blocks, trials_per_block = 3, 5
    reward_ratios = ((0.2, 0.8), (0.8, 0.2), (0.2, 0.8))
    plasticity_mask = np.zeros((3, 3, 3))
    plasticity_mask[1][1][0] = 1
    # assert len(reward_ratios) == num_blocks, print("length of reward ratios should be equal to number of blocks!")

    input_dim, output_dim = 2, 1
    key = jax.random.PRNGKey(0)

    fly_plasticity_coeff, plasticity_func = synapse.init_reward_volterra(init="reward")
    insilico_plasticity_coeff, _ = synapse.init_reward_volterra(init="zeros")

    key, key2 = split(key)

    # winit = jnp.zeros((input_dim, output_dim))

    winit = utils.generate_gaussian(key, (input_dim, output_dim), scale=0.01)
    print(f"initial weights: \n{winit}")
    key, key2 = split(key)

    num_odors = 2
    mus, sigmas = inputs.generate_binary_input_parameters()

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()

    start = time.time()
    xs, odors, decisions, rewards, exp_rewards = network.simulate_fly_experiment(
        key,
        winit,
        fly_plasticity_coeff,
        plasticity_func,
        mus,
        sigmas,
        reward_ratios,
        trials_per_block,
    )
    print("time taken to simulate experiment: ", time.time() - start)
    print()

    print("time taken to simulate fly: ", time.time() - start)

    rewards = np.array(rewards, dtype=float).flatten()
    exp_rewards = np.array(exp_rewards).flatten()
    xs = convert_xs_to_tensor(xs)
    decisions = convert_decisions_to_tensor(decisions)
    print("rewards: \n", rewards)
    print("exp rewards: \n", exp_rewards)
    # print("xs: \n", xs_tensor)
    print("decisions: \n", decisions)
    trial_lengths = np.sum(np.logical_not(np.isnan(decisions)), axis=1).astype(int)

    insilico_ys, _ = network.simulate_insilico_experiment(
        winit, fly_plasticity_coeff, plasticity_func, xs, rewards, exp_rewards, trial_lengths
    )
    print("insilico ys: \n", np.squeeze(insilico_ys))
    exit()

    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(insilico_plasticity_coeff)
    loss_t = []

    for epoch in range(num_epochs):
        start = time.time()
        loss, meta_grads = loss_value_and_grad(
            winit,
            insilico_plasticity_coeff,
            plasticity_func,
            xs,
            rewards,
            exp_rewards,
            fly_ys,
            plasticity_mask,
        )
        # check if loss is nan
        if np.isnan(loss):
            print("loss is nan!")
            break
        loss_t.append(loss)
        if epoch % 100 == 0:
            print(f"epoch :{epoch + 1}")
            print(f"loss :{loss}")
            print()

        updates, opt_state = optimizer.update(
            meta_grads, opt_state, insilico_plasticity_coeff
        )

        insilico_plasticity_coeff = optax.apply_updates(
            insilico_plasticity_coeff, updates
        )
        # jax.debug.print("insilico_plasticity_coeff: {}", insilico_plasticity_coeff)
        print()

    id_print(insilico_plasticity_coeff)

    # save loss into file as numpy array
    np.save("expdata/loss.npy", loss_t)
