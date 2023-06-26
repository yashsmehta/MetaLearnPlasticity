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

def convert_list_to_tensor(nested_list, list_type="decisions"):
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])

    num_trials = num_blocks * trials_per_block

    trial_lengths = [
                [len(nested_list[j][i]) for i in range(trials_per_block)]
                for j in range(num_blocks)
            ]

    longest_trial_length = np.max(np.array(trial_lengths))

    try: 
        element_dim = len(nested_list[0][0][0])
    except TypeError:  # if item is not iterable
        element_dim = 1 
    if list_type == "decisions" or list_type == "odors":
        # note: this has to be nan, since trial length is calculated by this.
        tensor = np.full((num_trials, longest_trial_length), np.nan)
    elif list_type == "xs":
        tensor = np.full((num_trials, longest_trial_length, element_dim), 0)
    else:
        raise Exception("type must be 'decisions' or 'xs'")

    for i in range(num_blocks):
        for j in range(trials_per_block):
            trial = nested_list[i][j]
            for k in range(len(trial)):
                tensor[i * trials_per_block + j][k] = trial[k]

    return tensor


if __name__ == "__main__":
    num_epochs = 3000
    num_blocks, trials_per_block = 3, 50
    reward_ratios = ((0.2, 0.8), (0.2, 0.8), (0.2, 0.8))
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
    print("experiment simulation time: ", time.time() - start)

    rewards = np.array(rewards, dtype=float).flatten()
    exp_rewards = np.array(exp_rewards).flatten()
    xs = convert_list_to_tensor(xs, list_type="xs")
    odors = convert_list_to_tensor(odors, list_type="odors")
    decisions = convert_list_to_tensor(decisions, list_type="decisions")
    trial_lengths = jnp.sum(jnp.logical_not(jnp.isnan(decisions)), axis=1).astype(int)

    mask = np.ones(decisions.shape)
    for i, length in enumerate(trial_lengths):
        mask[i][length:] = 0

    outputs, _ = network.simulate_insilico_experiment(
        winit, insilico_plasticity_coeff, plasticity_func, xs, rewards, exp_rewards, trial_lengths
    )

    # loss = losses.celoss(
    #     winit,
    #     insilico_plasticity_coeff,
    #     plasticity_func,
    #     xs,
    #     rewards,
    #     exp_rewards,
    #     decisions,
    #     trial_lengths,
    #     mask,
    #     plasticity_mask,
    # )
    # print("loss: ", loss)

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
            decisions,
            trial_lengths,
            mask,
            plasticity_mask,
        )
        # check if loss is nan
        if np.isnan(loss):
            print("loss is nan!")
            break
        loss_t.append(loss)
        if epoch % 100 == 0:
            print(f"iteration :{epoch}")
            print(f"loss :{loss}")
            id_print(insilico_plasticity_coeff[1][1][0])
            print()

        updates, opt_state = optimizer.update(
            meta_grads, opt_state, insilico_plasticity_coeff
        )

        insilico_plasticity_coeff = optax.apply_updates(
            insilico_plasticity_coeff, updates
        )
        # jax.debug.print("insilico_plasticity_coeff: {}", insilico_plasticity_coeff)


    # save loss into file as numpy array
    # np.save("expdata/loss.npy", loss_t)
