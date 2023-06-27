import plasticity.inputs as inputs
from plasticity import utils, synapse
import plasticity.behavior.network as network
import plasticity.behavior.losses as losses
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax.random import split
import time
from jax.experimental.host_callback import id_print
from pprint import pprint


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

    return jnp.array(tensor)


def simulate_all_experiments(
    key,
    num_exps,
    winit,
    fly_plasticity_coeff,
    plasticity_func,
    mus,
    sigmas,
    reward_ratios,
    trials_per_block,
):
    xs, odors, decisions, rewards, exp_rewards = {}, {}, {}, {}, {}

    for exp_i in range(num_exps):
        key, subkey = split(key)
        print(f"simulating experiment: {exp_i + 1}")
        (
            xs_exp,
            odors_exp,
            decisions_exp,
            rewards_exp,
            exp_rewards_exp,
        ) = network.simulate_fly_experiment(
            key,
            winit,
            fly_plasticity_coeff,
            plasticity_func,
            mus,
            sigmas,
            reward_ratios,
            trials_per_block,
        )

        xs[str(exp_i)] = convert_list_to_tensor(xs_exp, list_type="xs")
        odors[str(exp_i)] = convert_list_to_tensor(odors_exp, list_type="odors")
        decisions[str(exp_i)] = convert_list_to_tensor(
            decisions_exp, list_type="decisions"
        )
        rewards[str(exp_i)] = np.array(rewards_exp, dtype=float).flatten()
        exp_rewards[str(exp_i)] = np.array(exp_rewards_exp, dtype=float).flatten()

    return xs, odors, decisions, rewards, exp_rewards


if __name__ == "__main__":
    num_epochs = 1000
    num_exps = 10
    num_blocks, trials_per_block = 3, 100
    reward_ratios = ((0.2, 0.8), (0.8, 0.2), (0.2, 0.8))
    coeff_mask = np.zeros((3, 3, 3))
    coeff_mask[1,1,0] = 1
    coeff_mask[1,0,0] = 1
    coeff_mask[0,1,0] = 1
    coeff_mask[0,0,0] = 1

    assert len(reward_ratios) == num_blocks, print(
        "length of reward ratios should be equal to number of blocks!"
    )

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

    xs, odors, decisions, rewards, exp_rewards = simulate_all_experiments(
        key,
        num_exps,
        winit,
        fly_plasticity_coeff,
        plasticity_func,
        mus,
        sigmas,
        reward_ratios,
        trials_per_block,
    )
    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(insilico_plasticity_coeff)
    loss_all, coeffs_all = [], []

    for epoch in range(num_epochs):
        for exp_i in range(num_exps):
            start = time.time()
            trial_lengths = jnp.sum(
                jnp.logical_not(jnp.isnan(decisions[str(exp_i)])), axis=1
            ).astype(int)
            # print("trial lengths: \n", trial_lengths)
            # print("odors: \n", odors[str(exp_i)])

            logits_mask = np.ones(decisions[str(exp_i)].shape)
            for j, length in enumerate(trial_lengths):
                logits_mask[j][length:] = 0

            # print("logits mask: \n", logits_mask)

            # loss = losses.celoss(
            #     winit,
            #     insilico_plasticity_coeff,
            #     plasticity_func,
            #     xs[str(exp_i)],
            #     rewards[str(exp_i)],
            #     exp_rewards[str(exp_i)],
            #     decisions[str(exp_i)],
            #     trial_lengths,
            #     logits_mask,
            #     coeff_mask,
            # )

            loss, meta_grads = loss_value_and_grad(
                winit,
                insilico_plasticity_coeff,
                plasticity_func,
                xs[str(exp_i)],
                rewards[str(exp_i)],
                exp_rewards[str(exp_i)],
                decisions[str(exp_i)],
                trial_lengths,
                logits_mask,
                coeff_mask,
            )

            updates, opt_state = optimizer.update(
                meta_grads, opt_state, insilico_plasticity_coeff
            )

            insilico_plasticity_coeff = optax.apply_updates(
                insilico_plasticity_coeff, updates
            )
        # check if loss is nan
        if np.isnan(loss):
            print("loss is nan!")
            break
        if epoch % 100 == 0:
            print(f"epoch :{epoch}")
            print(f"loss :{loss}")
            print(
                insilico_plasticity_coeff[1,1,0],
                insilico_plasticity_coeff[1,0,0],
                insilico_plasticity_coeff[0,1,0],
                insilico_plasticity_coeff[0,0,0],
            )
            print()
            coeffs_all.append(insilico_plasticity_coeff[1,1,0])
            loss_all.append(loss)
        # jax.debug.print("insilico_plasticity_coeff: {}", insilico_plasticity_coeff)

    # save loss into file as numpy array
    # np.save("expdata/loss.npy", np.array(loss_t))
    # np.save("expdata/coeff.npy", np.array(coeff_t))
