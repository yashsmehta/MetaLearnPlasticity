import jax.numpy as jnp
import jax
from jax import vmap
from jax.lax import reshape
from plasticity import inputs
from jax.random import bernoulli, split
from functools import partial
import numpy as np
from plasticity.utils import create_nested_list


def simulate_fly_trial(key, weights, plasticity_coeffs, plasticity_func, reward_in_arena, r_history, odor_mus, odor_sigmas):
    input_xs, trial_odors, sampled_outputs = [], [], []
    moving_avg_window = 10
    
    while True:
        key, subkey = split(key)
        odor = int(bernoulli(key, 0.5))
        trial_odors.append(odor)
        x = inputs.sample_inputs(odor_mus, odor_sigmas, odor, subkey)
        prob_output = jax.nn.sigmoid(jnp.dot(x, weights))
        key, subkey = split(key)
        sampled_output = float(bernoulli(subkey, prob_output))

        input_xs.append(x)
        sampled_outputs.append(sampled_output)

        if sampled_output == 0:
            reward = reward_in_arena[odor]
            reward_in_arena[odor] = 0
            exp_reward = sum(r_history[-moving_avg_window:]) / moving_avg_window
            dw = weight_update(x, weights, plasticity_coeffs, plasticity_func, reward, exp_reward)
            weights += dw
            r_history.append(reward)
            break

    return (input_xs, trial_odors, sampled_outputs, reward, exp_reward), weights, reward_in_arena, r_history


def simulate_fly_experiment(
    key,
    weights,
    plasticity_coeffs,
    plasticity_func,
    odor_mus,
    odor_sigmas,
    reward_ratios,
    trials_per_block
):
    num_blocks = len(reward_ratios)
    # we need to keep a track of this to calculate the expected reward for odors
    r_history = []
    reward_in_arena = np.zeros(2,)

    xs, odors, sampled_ys, rewards, exp_rewards = (
        create_nested_list(num_blocks, trials_per_block) for _ in range(5)
    )

    for block in range(num_blocks):
        r_ratio = reward_ratios[block]
        for trial in range(trials_per_block):
            key, key2 = split(key)
            if reward_in_arena[0] == 0:
                reward_in_arena[0] = bernoulli(key, r_ratio[0])
            if reward_in_arena[1] == 0:
                reward_in_arena[1] = bernoulli(key2, r_ratio[1])

            key, _ = split(key)
            trial_data, weights, reward_in_area, r_history = simulate_fly_trial(
                key,
                weights,
                plasticity_coeffs,
                plasticity_func,
                reward_in_arena,
                r_history,
                odor_mus,
                odor_sigmas,
            )
            (
                xs[block][trial],
                odors[block][trial],
                sampled_ys[block][trial],
                rewards[block][trial],
                exp_rewards[block][trial],
            ) = trial_data

    return xs, odors, sampled_ys, rewards, exp_rewards


@partial(jax.jit, static_argnames=["plasticity_func"])
def simulate_insilico_trial(weights, plasticity_coeffs, plasticity_func, x, reward, exp_reward):

    outputs = [jax.nn.sigmoid(jnp.dot(xi, weights)) for xi in x]
    # call the weight update with the last input of the trial
    dw = weight_update(x[-1], weights, plasticity_coeffs, plasticity_func, reward, exp_reward)
    weights += dw

    return outputs, weights


def simulate_insilico_experiment(
    weights,
    plasticity_coeffs,
    plasticity_func,
    xs,
    rewards,
    exp_rewards
):
    num_blocks, trials_per_block = len(xs), len(xs[0])

    ys = create_nested_list(num_blocks, trials_per_block)

    for block in range(num_blocks):
        for trial in range(trials_per_block):
            ys[block][trial], weights = simulate_insilico_trial(
                weights,
                plasticity_coeffs,
                plasticity_func,
                xs[block][trial],
                rewards[block][trial],
                exp_rewards[block][trial]
            )

    return ys

def weight_update(
    x, weights, plasticity_coeffs, plasticity_func, reward, exp_reward
):
    reward_term = reward - exp_reward
    # reward_term = reward
    m, n = weights.shape
    in_grid, _ = jnp.meshgrid(
        reshape(x, (m,)),
        jnp.ones(
            n,
        ),
        indexing="ij",
    )

    vfun = vmap(plasticity_func, in_axes=(0, None, 0, None))
    dw = vmap(vfun, in_axes=(1, None, 1, None), out_axes=1)(
        in_grid, reward_term, weights, plasticity_coeffs
    )

    assert (
        dw.shape == weights.shape
    ), "dw and w should be of the same shape to prevent broadcasting while adding"

    return dw
