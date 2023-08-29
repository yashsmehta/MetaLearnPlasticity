import jax.numpy as jnp
import jax
from jax import vmap
from functools import partial
import plasticity.behavior.data_loader as data_loader
import plasticity.behavior.utils as utils

@jax.jit
def network_forward(params, inputs):
    """Forward pass for the network
    Returns:
        activations for all layers, and logits
    """
    print("compiling model.network_forward()...")
    activations = [inputs]
    activation = inputs
    for w, b in params[:-1]:
        activation = jax.nn.sigmoid(activation @ w + b)
        activations.append(activation)

    final_w, final_b = params[-1]
    logits = activation @ final_w + final_b
    activations.append(logits)
    return activations


def simulate(
    initial_params,
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
        a tensor of activations for the experiment, and the params_trajec,
        i.e. the params at each trial.
        shapes:
            activations: [(num_trials, trial_length)
            weight tensor: (num_trials, input_dim, output_dim)
    """

    print("compiling model.simulate()...")

    def step(carry, stimulus):
        params = carry
        x, reward, expected_reward, trial_length = stimulus
        params, activation = network_step(
            x,
            params,
            plasticity_coeffs,
            plasticity_func,
            reward,
            expected_reward,
            trial_length,
        )
        return params, activation

    final_params, (params_trajec, activations) = jax.lax.scan(
        step, initial_params, (xs, rewards, expected_rewards, trial_lengths)
    )
    return params_trajec, activations


def network_step(
    input,
    params,
    plasticity_coeffs,
    plasticity_func,
    reward,
    expected_reward,
    trial_length,
):
    """Performs a forward pass and weight update
        Forward pass is needed to compute logits for the loss function
    Returns:
        updated params, and stacked: params, logits
    """
    activations = jax.vmap(network_forward, in_axes=(None, 0))(params, input)
    # pass only the activations wrt the last odor in trial
    activations = [a[trial_length - 1] for a in activations]
    params = update_params(
        params, activations, plasticity_coeffs, plasticity_func, reward, expected_reward
    )

    return params, (params, activations)


@partial(jax.jit, static_argnums=(3,))
def update_params(
    params, activations, plasticity_coeffs, plasticity_func, reward, expected_reward
):
    """assuming plasticity happens in the first layer only.
    [dw, db] = plasticity_func(activation, reward, w, plasticity_coeffs)
    returns updated params
    """
    print("compiling model.update_params()...")
    reward_term = reward - expected_reward

    delta_params = []

    # plasticity happens in the first layer only
    activation = activations[0]
    w, b = params[0]
    # vmap over output neurons
    vmap_inputs = jax.vmap(plasticity_func, in_axes=(None, None, 0, None))
    # vmap over input neurons
    vmap_synapses = jax.vmap(vmap_inputs, in_axes=(0, None, 0, None))
    dw = vmap_synapses(activation, reward_term, w, plasticity_coeffs)
    db = vmap_inputs(1.0, reward_term, b, plasticity_coeffs)
    assert (
        dw.shape == w.shape and db.shape == b.shape
    ), "dw and w should be of the same shape to prevent broadcasting \
        while adding"
    delta_params.append((dw, db))

    # add the last layer of no plasticity
    if len(params) > len(delta_params):
        delta_params.append((0.0, 0.0))
    params = [(w + dw, b + db) for (w, b), (dw, db) in zip(params, delta_params)]

    return params


def evaluate(
    key, cfg, simulation_coeff, plasticity_coeff, plasticity_func, mus, sigmas
):
    """Evaluate logits, weight trajectory for simulation_coeff and plasticity_coeff
       with new initial params, for a single new experiment
    Returns:
        logits, model_logits: (total_trials, longest_trial),
        params_trajec, model_weight_trajec: (total_trials, input_dim, output_dim)
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

    logits, params_trajec = simulate(
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
    return (logits, params_trajec), (model_logits, model_weight_trajec)
