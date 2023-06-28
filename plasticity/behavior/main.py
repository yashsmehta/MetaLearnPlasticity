import plasticity.inputs as inputs
from plasticity import utils, synapse
import plasticity.behavior.data_loader as data_loader
import plasticity.behavior.losses as losses
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax.random import split
import time


if __name__ == "__main__":
    num_epochs = 1000
    num_exps = 10
    num_blocks, trials_per_block = 3, 100
    num_odors = 2
    reward_ratios = ((0.2, 0.8), (0.8, 0.2), (0.2, 0.8))
    # specify trainable parameters
    coeff_mask = np.zeros((3, 3, 3))
    coeff_mask[1, 1, 0] = 1
    coeff_mask[1, 0, 0] = 1
    coeff_mask[0, 1, 0] = 1
    coeff_mask[0, 0, 0] = 1

    assert len(reward_ratios) == num_blocks, print(
        "length of reward ratios should be equal to number of blocks!"
    )

    input_dim, output_dim = 2, 1
    key = jax.random.PRNGKey(0)

    simulation_coeff, plasticity_func = synapse.init_reward_volterra(init="reward")
    plasticity_coeff, _ = synapse.init_reward_volterra(init="zeros")

    key, key2 = split(key)

    # winit = jnp.zeros((input_dim, output_dim))
    winit = utils.generate_gaussian(key, (input_dim, output_dim), scale=0.01)
    print(f"initial weights: \n{winit}")
    key, key2 = split(key)

    mus, sigmas = inputs.generate_binary_input_parameters()

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(input_dim, output_dim))
    print()

    (
        xs,
        odors,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.simulate_all_experiments(
        key,
        num_exps,
        winit,
        simulation_coeff,
        plasticity_func,
        mus,
        sigmas,
        reward_ratios,
        trials_per_block,
    )

    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(plasticity_coeff)

    for epoch in range(num_epochs):
        for exp_i in range(num_exps):
            start = time.time()
            # calculate the length of each trial by checking for NaNs
            trial_lengths = jnp.sum(
                jnp.logical_not(jnp.isnan(decisions[str(exp_i)])), axis=1
            ).astype(int)

            logits_mask = np.ones(decisions[str(exp_i)].shape)
            for j, length in enumerate(trial_lengths):
                logits_mask[j][length:] = 0

            # debug
            # print("trial lengths: \n", trial_lengths)
            # print("odors: \n", odors[str(exp_i)])
            # print("logits mask: \n", logits_mask)
            # loss = losses.celoss(
            #     winit,
            #     plasticity_coeff,
            #     plasticity_func,
            #     xs[str(exp_i)],
            #     rewards[str(exp_i)],
            #     expected_rewards[str(exp_i)],
            #     decisions[str(exp_i)],
            #     trial_lengths,
            #     logits_mask,
            #     coeff_mask,
            # )
            # exit()

            loss, meta_grads = loss_value_and_grad(
                winit,
                plasticity_coeff,
                plasticity_func,
                xs[str(exp_i)],
                rewards[str(exp_i)],
                expected_rewards[str(exp_i)],
                decisions[str(exp_i)],
                trial_lengths,
                logits_mask,
                coeff_mask,
            )

            updates, opt_state = optimizer.update(
                meta_grads, opt_state, plasticity_coeff
            )

            plasticity_coeff = optax.apply_updates(plasticity_coeff, updates)

        # check if loss is nan
        if np.isnan(loss):
            print("loss is nan!")
            break
        if epoch % 100 == 0:
            print(f"epoch :{epoch}")
            print(f"loss :{loss}")
            print(
                plasticity_coeff[1, 1, 0],
                plasticity_coeff[1, 0, 0],
                plasticity_coeff[0, 1, 0],
                plasticity_coeff[0, 0, 0],
            )
            # print(
            #     plasticity_coeff
            # )
            print()

        # jax.debug.print("plasticity_coeff: {}", plasticity_coeff)

    # save loss into file as numpy array
    # np.save("expdata/loss.npy", np.array(loss_t))
    # np.save("expdata/coeff.npy", np.array(coeff_t))
