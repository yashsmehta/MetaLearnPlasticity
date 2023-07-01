import jax.numpy as jnp
import jax
from jax import vmap
from jax.lax import reshape
import plasticity.behavior.data_loader as data_loader
import plasticity.behavior.utils as utils


def simulate(
    initial_weights,
    plasticity_coeffs,
    plasticity_func,
    xs,
    rewards,
    expected_rewards,
    trial_lengths,
):
    """Simulate an experiment with given plasticity coefficients,
       vmap over timesteps within a trial, and scan over all trials

    Returns:
        a tensor of logits for the experiment, and the weight_trajec,
        i.e. the weights at each trial.
        shapes:
            logits: (total_trials, max_trial_length)
            weight tensor: (total_trials, input_dim, output_dim)
    """

    def step(weights, stimulus):
        x, reward, expected_reward, trial_length = stimulus
        return network_step(
            x,
            weights,
            plasticity_coeffs,
            plasticity_func,
            reward,
            expected_reward,
            trial_length,
        )

    final_weights, (weight_trajec, logits) = jax.lax.scan(
        step, initial_weights, (xs, rewards, expected_rewards, trial_lengths)
    )
    return jnp.squeeze(logits), weight_trajec


def network_step(
    input,
    weights,
    plasticity_coeffs,
    plasticity_func,
    reward,
    expected_reward,
    trial_length,
):
    """Performs a forward pass and weight update
        Forward pass is needed to compute logits for the loss function
    Returns:
        updated weights, and stacked: weights, logits
    """
    vmapped_forward = vmap(lambda weights, x: jnp.dot(x, weights), (None, 0))
    logits = vmapped_forward(weights, input)
    x = input[trial_length - 1]
    dw = weight_update(
        x, weights, plasticity_coeffs, plasticity_func, reward, expected_reward
    )
    weights += dw

    return weights, (weights, logits)


def weight_update(
    x, weights, plasticity_coeffs, plasticity_func, reward, expected_reward
):
    """Weight update for all synapses in the layer. Uses vmap to vectorize
       plasticity function over all synapses
    Returns:
        delta weights for the layer
    """
    reward_term = reward - expected_reward
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
    ), "dw and w should be of the same shape to prevent broadcasting \
        while adding"

    return dw


def evaluate(
    key, cfg, simulation_coeff, plasticity_coeff, plasticity_func, mus, sigmas
):
    """Evaluate logits, weight trajectory for simulation_coeff and plasticity_coeff
       with new initial weights, for a single new experiment
    Returns:
        logits, model_logits: (total_trials, longest_trial), 
        weight_trajec, model_weight_trajec: (total_trials, input_dim, output_dim)
    """

    test_cfg = cfg.copy()
    test_cfg.num_exps = 1
    winit = utils.generate_gaussian(key, (cfg.input_dim, cfg.output_dim), scale=0.01)

    (
        xs,
        odors,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.generate_experiments_data(
        key,
        test_cfg,
        winit,
        simulation_coeff,
        plasticity_func,
        mus,
        sigmas,
    )
    trial_lengths = jnp.sum(jnp.logical_not(jnp.isnan(decisions["0"])), axis=1).astype(
        int
    )

    logits, weight_trajec = simulate(
        winit,
        simulation_coeff,
        plasticity_func,
        xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )

    model_logits, model_weight_trajec = simulate(
        winit,
        plasticity_coeff,
        plasticity_func,
        xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )
    return (logits, weight_trajec), (model_logits, model_weight_trajec)
