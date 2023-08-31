import numpy as np
import jax
from jax.random import bernoulli, split
from jax.nn import sigmoid
import collections
import scipy.io as sio
import os
from functools import partial

import plasticity.behavior.model as model
from plasticity.behavior.utils import experiment_list_to_tensor
from plasticity.behavior.utils import create_nested_list
from plasticity import inputs


def generate_experiments_data(
    key,
    cfg,
    params,
    plasticity_coeff,
    plasticity_func,
    odor_mus,
    odor_sigmas,
):

    """Simulate all fly experiments with given plasticity coefficients
    Returns:
        5 dictionaries corresponding with experiment number (int) as key, with
        tensors as values
    """

    xs, odors, decisions, rewards, expected_rewards = {}, {}, {}, {}, {}
    print("generating experiments data...")

    for exp_i in range(cfg.num_exps):
        key, _ = split(key)
        print(f"simulating experiment: {exp_i + 1}")
        (
            exp_xs,
            exp_odors,
            exp_decisions,
            exp_rewards,
            exp_expected_rewards,
        ) = generate_experiment(
            key,
            cfg,
            params,
            plasticity_coeff,
            plasticity_func,
            odor_mus,
            odor_sigmas,
        )

        trial_lengths = [
            [len(exp_decisions[j][i]) for i in range(cfg.trials_per_block)]
            for j in range(cfg.num_blocks)
        ]
        longest_trial_length = np.max(np.array(trial_lengths))
        print(f"longest trial length: {longest_trial_length}")

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


def generate_experiment(
    key,
    cfg,
    params,
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

            trial_data, params, rewards_in_arena, r_history = generate_trial(
                key,
                params,
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


def generate_trial(
    key,
    params,
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
        x = inputs.sample_inputs(key, odor_mus, odor_sigmas, odor)
        resampled_x = inputs.sample_inputs(subkey, odor_mus, odor_sigmas, odor)
        jit_network_forward = jax.jit(model.network_forward)
        activations = jit_network_forward(params, x)
        prob_output = sigmoid(activations[-1])

        key, subkey = split(key)
        sampled_output = float(bernoulli(subkey, prob_output))

        input_xs.append(resampled_x)
        decisions.append(sampled_output)

        if sampled_output == 1:
            reward = rewards_in_arena[odor]
            r_history.appendleft(reward)
            rewards_in_arena[odor] = 0
            jit_update_params = partial(jax.jit, static_argnums=(3,))(model.update_params)
            params = jit_update_params(
                params, activations, plasticity_coeffs, plasticity_func, reward, expected_reward
            )
            break

    return (
        (input_xs, trial_odors, decisions, reward, expected_reward),
        params,
        rewards_in_arena,
        r_history,
    )

def expected_reward_for_exp_data(R, moving_avg_window):
    r_history = collections.deque(moving_avg_window * [0], maxlen=moving_avg_window)
    expected_rewards = []
    for r in R:
        expected_rewards.append(np.mean(r_history))
        r_history.appendleft(r)
    return np.array(expected_rewards)


def load_adi_expdata(cfg):
    assert cfg.num_exps <= len(os.listdir(cfg.data_dir)), "Not enough experimental data"
    print("Loading experimental data...")
    
    element_dim = 2

    xs, decisions, rewards, expected_rewards = {}, {}, {}, {}

    for exp_i, file in enumerate(os.listdir(cfg.data_dir)):
        if exp_i >= cfg.num_exps:
            break
        print(exp_i, file)
        data = sio.loadmat(cfg.data_dir + file)
        X, Y, R = data["X"], data["Y"], data["R"]
        Y = np.squeeze(Y)
        R = np.squeeze(R)
        num_trials = np.sum(Y)
        assert num_trials == R.shape[0], "Y and R should have the same number of trials"

        # remove last element, and append left to get indices.
        indices = np.cumsum(Y)
        indices = np.insert(indices, 0, 0)
        indices = np.delete(indices, -1)

        exp_decisions = [[] for _ in range(num_trials)]
        exp_xs = [[] for _ in range(num_trials)]

        for index, decision, x in zip(indices, Y, X):
            exp_decisions[index].append(decision)
            exp_xs[index].append(x)

        trial_lengths = [len(exp_decisions[i]) for i in range(num_trials)]
        longest_trial_length = np.max(np.array(trial_lengths))

        d_tensor = np.full((num_trials, longest_trial_length), np.nan)
        for i in range(num_trials):
            for j in range(trial_lengths[i]):
                d_tensor[i][j] = exp_decisions[i][j]
        decisions[str(exp_i)] = d_tensor

        xs_tensor = np.full((num_trials, longest_trial_length, element_dim), 0)
        for i in range(num_trials):
            for j in range(trial_lengths[i]):
                xs_tensor[i][j] = exp_xs[i][j]
        xs[str(exp_i)] = xs_tensor

        rewards[str(exp_i)] = R
        expected_rewards[str(exp_i)] = expected_reward_for_exp_data(R, cfg.moving_avg_window)

    return xs, decisions, rewards, expected_rewards


def load_single_adi_experiment(cfg):
    
    exp_i = 0
    element_dim = 2

    xs, decisions, rewards, expected_rewards = {}, {}, {}, {}
    file = f"Fly{cfg.jobid}.mat"
    data = sio.loadmat(cfg.data_dir + file)
    X, Y, R = data["X"], data["Y"], data["R"]
    Y = np.squeeze(Y)
    R = np.squeeze(R)
    num_trials = np.sum(Y)
    assert num_trials == R.shape[0], "Y and R should have the same number of trials"

    # remove last element, and append left to get indices.
    indices = np.cumsum(Y)
    indices = np.insert(indices, 0, 0)
    indices = np.delete(indices, -1)

    exp_decisions = [[] for _ in range(num_trials)]
    exp_xs = [[] for _ in range(num_trials)]

    for index, decision, x in zip(indices, Y, X):
        exp_decisions[index].append(decision)
        exp_xs[index].append(x)

    trial_lengths = [len(exp_decisions[i]) for i in range(num_trials)]
    longest_trial_length = np.max(np.array(trial_lengths))

    d_tensor = np.full((num_trials, longest_trial_length), np.nan)
    for i in range(num_trials):
        for j in range(trial_lengths[i]):
            d_tensor[i][j] = exp_decisions[i][j]
    decisions[str(exp_i)] = d_tensor

    xs_tensor = np.full((num_trials, longest_trial_length, element_dim), 0)
    for i in range(num_trials):
        for j in range(trial_lengths[i]):
            xs_tensor[i][j] = exp_xs[i][j]
    xs[str(exp_i)] = xs_tensor

    rewards[str(exp_i)] = R
    expected_rewards[str(exp_i)] = expected_reward_for_exp_data(R, cfg.moving_avg_window)

    return xs, decisions, rewards, expected_rewards