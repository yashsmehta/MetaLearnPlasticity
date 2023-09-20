import numpy as np
import jax
from jax.random import bernoulli, split
from jax.nn import sigmoid
import jax.numpy as jnp
import collections
import scipy.io as sio
import os
from functools import partial

import plasticity.model as model
import plasticity.inputs as inputs
import plasticity.synapse as synapse
from plasticity.utils import experiment_list_to_tensor
from plasticity.utils import create_nested_list


def load_data(key, cfg):
    if cfg.use_experimental_data:
        return load_adi_expdata(key, cfg)
    
    else:
        generation_coeff, generation_func = synapse.init_plasticity(
            key, cfg, mode="generation_model"
        )
        return generate_experiments_data(key, cfg, generation_coeff, generation_func)

def generate_experiments_data(
    key,
    cfg,
    plasticity_coeff,
    plasticity_func,
):
    """Simulate all fly experiments with given plasticity coefficients
    Returns:
        5 dictionaries corresponding with experiment number (int) as key, with
        tensors as values
    """

    xs, odors, neural_recordings, decisions, rewards, expected_rewards = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    print("generating experiments data...")

    for exp_i in range(cfg.num_exps):
        seed = (cfg.jobid + 1) * (exp_i + 1)
        odor_mus, odor_sigmas = inputs.generate_input_parameters(seed, cfg)
        exp_i = str(exp_i)
        key, subkey = split(key)
        params = model.initialize_params(key, cfg)
        # print("prob_output:")
        (
            exp_xs,
            exp_odors,
            exp_neural_recordings,
            exp_decisions,
            exp_rewards,
            exp_expected_rewards,
        ) = generate_experiment(
            subkey,
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
        max_trial_length = np.max(np.array(trial_lengths))
        print("Exp " + exp_i + f", longest trial length: {max_trial_length}")

        xs[exp_i] = experiment_list_to_tensor(max_trial_length, exp_xs, list_type="xs")
        odors[exp_i] = experiment_list_to_tensor(
            max_trial_length, exp_odors, list_type="odors"
        )
        neural_recordings[exp_i] = experiment_list_to_tensor(
            max_trial_length, exp_neural_recordings, list_type="neural_recordings"
        )
        decisions[exp_i] = experiment_list_to_tensor(
            max_trial_length, exp_decisions, list_type="decisions"
        )
        rewards[exp_i] = np.array(exp_rewards, dtype=float).flatten()
        expected_rewards[exp_i] = np.array(exp_expected_rewards, dtype=float).flatten()
        # print("odors: ", odors[exp_i])
        # print("rewards: ", rewards[exp_i])

    return xs, neural_recordings, decisions, rewards, expected_rewards


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

    xs, odors, neural_recordings, decisions, rewards, expected_rewards = (
        create_nested_list(cfg.num_blocks, cfg.trials_per_block) for _ in range(6)
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
                neural_recordings[block][trial],
                decisions[block][trial],
                rewards[block][trial],
                expected_rewards[block][trial],
            ) = trial_data

    return xs, odors, neural_recordings, decisions, rewards, expected_rewards


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

    input_xs, trial_odors, neural_recordings, decisions = [], [], [], []

    expected_reward = np.mean(r_history)

    while True:
        key, _ = split(key)
        odor = int(bernoulli(key, 0.5))
        trial_odors.append(odor)
        key, subkey = split(key)
        x = inputs.sample_inputs(key, odor_mus, odor_sigmas, odor)
        resampled_x = inputs.sample_inputs(subkey, odor_mus, odor_sigmas, odor)
        jit_network_forward = jax.jit(model.network_forward)
        activations = jit_network_forward(params, x)
        prob_output = sigmoid(activations[-1])

        key, subkey = split(key)
        sampled_output = float(bernoulli(subkey, prob_output))

        input_xs.append(resampled_x)
        neural_recordings.append(activations[-2])
        decisions.append(sampled_output)

        if sampled_output == 1:
            # print(prob_output)
            reward = rewards_in_arena[odor]
            r_history.appendleft(reward)
            rewards_in_arena[odor] = 0
            jit_update_params = partial(jax.jit, static_argnums=(3,))(
                model.update_params
            )
            params = jit_update_params(
                params,
                activations,
                plasticity_coeffs,
                plasticity_func,
                reward,
                expected_reward,
            )
            break

    return (
        (input_xs, trial_odors, neural_recordings, decisions, reward, expected_reward),
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


def load_adi_expdata(key, cfg):
    print("Loading experimental data...")

    xs, neural_recordings, decisions, rewards, expected_rewards = {}, {}, {}, {}, {}

    input_dim = cfg.layer_sizes[0]
    for exp_i, file in enumerate(os.listdir(cfg.data_dir)):
        seed = (cfg.jobid + 1) * (exp_i + 1)
        key, _ = split(key)
        if exp_i >= cfg.num_exps:
            break
        odor_mus, odor_sigmas = inputs.generate_input_parameters(seed, cfg)
        exp_i = str(exp_i)
        print(exp_i, file)
        data = sio.loadmat(cfg.data_dir + file)
        X, Y, R = data["X"], data["Y"], data["R"]
        odors = np.where(X == 1)[1]
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

        for index, decision, odor in zip(indices, Y, odors):
            exp_decisions[index].append(decision)
            x = inputs.sample_inputs(key, odor_mus, odor_sigmas, odor)
            exp_xs[index].append(x)

        trial_lengths = [len(exp_decisions[i]) for i in range(num_trials)]
        max_trial_length = np.max(np.array(trial_lengths))

        d_tensor = np.full((num_trials, max_trial_length), np.nan)
        for i in range(num_trials):
            for j in range(trial_lengths[i]):
                d_tensor[i][j] = exp_decisions[i][j]
        decisions[exp_i] = d_tensor

        xs_tensor = np.full((num_trials, max_trial_length, input_dim), 0.)
        for i in range(num_trials):
            for j in range(trial_lengths[i]):
                xs_tensor[i][j] = exp_xs[i][j]
        xs[exp_i] = xs_tensor

        rewards[exp_i] = R
        expected_rewards[exp_i] = expected_reward_for_exp_data(R, cfg.moving_avg_window)
        neural_recordings[str(exp_i)] = None

    return xs, neural_recordings, decisions, rewards, expected_rewards


def get_trial_lengths(decisions):
    trial_lengths = jnp.sum(jnp.logical_not(jnp.isnan(decisions)), axis=1).astype(int)
    return trial_lengths


def get_logits_mask(decisions):
    logits_mask = jnp.logical_not(jnp.isnan(decisions)).astype(int)
    return logits_mask
