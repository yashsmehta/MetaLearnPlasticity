import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap
from jax.lax import reshape
from jax.random import bernoulli, split
from functools import partial
from jax.nn import sigmoid
from jax.experimental.host_callback import id_print
import collections

import plasticity.behavior.model as model
from plasticity.behavior.utils import experiment_list_to_tensor
from plasticity.behavior.utils import create_nested_list
from plasticity import inputs


def simulate_all_experiments(
    key,
    num_exps,
    winit,
    plasticity_coeff,
    plasticity_func,
    mus,
    sigmas,
    reward_ratios,
    trials_per_block,
):

    """Simulate all fly experiments with given plasticity coefficients
    Returns:
        5 dictionaries corresponding with experiment number (int) as key, with
        tensors as values
    """

    num_blocks = len(reward_ratios)
    xs, odors, decisions, rewards, expected_rewards = {}, {}, {}, {}, {}

    for exp_i in range(num_exps):
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
            winit,
            plasticity_coeff,
            plasticity_func,
            mus,
            sigmas,
            reward_ratios,
            trials_per_block,
        )

        trial_lengths = [
            [len(exp_decisions[j][i]) for i in range(trials_per_block)]
            for j in range(num_blocks)
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
    weights,
    plasticity_coeffs,
    plasticity_func,
    odor_mus,
    odor_sigmas,
    reward_ratios,
    trials_per_block,
    moving_avg_window=10,
):
    """Simulate a single fly experiment with given plasticity coefficients
    Returns:
        a nested list (num_blocks x trials_per_block) of lists of different
        lengths corresponding to the number of timesteps in each trial
    """

    num_blocks = len(reward_ratios)
    r_history = collections.deque(moving_avg_window * [0], maxlen=moving_avg_window)
    rewards_in_arena = np.zeros(
        2,
    )

    xs, odors, sampled_ys, rewards, expected_rewards = (
        create_nested_list(num_blocks, trials_per_block) for _ in range(5)
    )

    for block in range(len(reward_ratios)):
        r_ratio = reward_ratios[block]
        for trial in range(trials_per_block):
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
        a tuple containing lists of xs, odors, decisions (sampled outputs), rewards, 
        and expected_rewards for the trial
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
