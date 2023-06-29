import numpy as np
import jax.numpy as jnp
from jax.random import bernoulli, split
from jax.nn import sigmoid
import collections

import plasticity.behavior.model as model
from plasticity.behavior.utils import experiment_list_to_tensor
from plasticity.behavior.utils import create_nested_list
from plasticity import inputs


def simulate_all_experiments(
    key,
    cfg,
    winit,
    plasticity_coeff,
    plasticity_func,
    mus,
    sigmas,
):

    """Simulate all fly experiments with given plasticity coefficients
    Returns:
        5 dictionaries corresponding with experiment number (int) as key, with
        tensors as values
    """

    xs, odors, decisions, rewards, expected_rewards = {}, {}, {}, {}, {}

    for exp_i in range(cfg.num_exps):
        key, subkey = split(key)
        print(f"simulating experiment: {exp_i + 1}")
        (
            exp_xs,
            exp_odors,
            exp_decisions,
            exp_rewards,
            exp_expected_rewards,
        ) = simulate_fly_experiment(
            key,
            cfg,
            winit,
            plasticity_coeff,
            plasticity_func,
            mus,
            sigmas,
        )

        trial_lengths = [
            [len(exp_decisions[j][i]) for i in range(cfg.trials_per_block)]
            for j in range(cfg.num_blocks)
        ]
        longest_trial_length = np.max(np.array(trial_lengths))

        xs[str(exp_i)] = experiment_list_to_tensor(
            longest_trial_length, exp_xs, list_type="xs"
        )
        odors[str(exp_i)] = experiment_list_to_tensor(
            longest_trial_length, exp_odors, list_type="odors"
        )
        decisions[str(exp_i)] = experiment_list_to_tensor(
            longest_trial_length, exp_decisions, list_type="decisions"
        )
        rewards[str(exp_i)] = np.array(exp_rewards, dtype=float).flatten()
        expected_rewards[str(exp_i)] = np.array(
            exp_expected_rewards, dtype=float
        ).flatten()

    return xs, odors, decisions, rewards, expected_rewards


def simulate_fly_experiment(
    key,
    cfg,
    weights,
    plasticity_coeffs,
    plasticity_func,
    odor_mus,
    odor_sigmas,
):
    """Simulate a single fly experiment with given plasticity coefficients
    Returns:
        a nested list (num_blocks x trials_per_block) of lists of different
        lengths corresponding to the number of timesteps in each trial
    """

    r_history = collections.deque(
        cfg.moving_avg_window * [0], maxlen=cfg.moving_avg_window
    )
    rewards_in_arena = np.zeros(
        2,
    )

    xs, odors, sampled_ys, rewards, expected_rewards = (
        create_nested_list(cfg.num_blocks, cfg.trials_per_block) for _ in range(5)
    )

    for block in range(cfg.num_blocks):
        r_ratio = cfg.reward_ratios[block]
        for trial in range(cfg.trials_per_block):
            key, _ = split(key)
            sampled_rewards = bernoulli(key, np.array(r_ratio))
            rewards_in_arena = np.logical_or(sampled_rewards, rewards_in_arena)
            key, _ = split(key)

            trial_data, weights, rewards_in_arena, r_history = simulate_fly_trial(
                key,
                weights,
                plasticity_coeffs,
                plasticity_func,
                rewards_in_arena,
                r_history,
                odor_mus,
                odor_sigmas,
            )
            (
                xs[block][trial],
                odors[block][trial],
                sampled_ys[block][trial],
                rewards[block][trial],
                expected_rewards[block][trial],
            ) = trial_data

    return xs, odors, sampled_ys, rewards, expected_rewards


def simulate_fly_trial(
    key,
    weights,
    plasticity_coeffs,
    plasticity_func,
    rewards_in_arena,
    r_history,
    odor_mus,
    odor_sigmas,
):
    """Simulate a single fly trial, which ends when the fly accepts odor
    Returns:
        a tuple containing lists of xs, odors, decisions (sampled outputs),
        rewards, and expected_rewards for the trial
    """

    input_xs, trial_odors, decisions = [], [], []

    expected_reward = np.mean(r_history)

    while True:
        key, subkey = split(key)
        odor = int(bernoulli(key, 0.5))
        trial_odors.append(odor)
        x = inputs.sample_inputs(odor_mus, odor_sigmas, odor, subkey)
        prob_output = sigmoid(jnp.dot(x, weights))
        key, subkey = split(key)
        sampled_output = float(bernoulli(subkey, prob_output))

        input_xs.append(x)
        decisions.append(sampled_output)

        if sampled_output == 1:
            reward = rewards_in_arena[odor]
            r_history.appendleft(reward)
            rewards_in_arena[odor] = 0
            dw = model.weight_update(
                x, weights, plasticity_coeffs, plasticity_func, reward, expected_reward
            )
            weights += dw
            break

    return (
        (input_xs, trial_odors, decisions, reward, expected_reward),
        weights,
        rewards_in_arena,
        r_history,
    )
