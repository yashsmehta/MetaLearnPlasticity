import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
import plasticity.behavior.data_loader as data_loader
import plasticity.behavior.utils as utils
import plasticity.synapse as synapse


def initialize_params(key, cfg, scale=0.01):
    """
    Initialize parameters for the network;
    There is no plasticity in the
    Returns:
        list of tuples of weights and biases for each layer
    """
    layer_sizes = cfg.layer_sizes
    output_dim = layer_sizes[-1]
    assert output_dim == 1, "output_dim should be 1, as that is the prob(choose odor)"

    initial_params = [
        (
            utils.generate_gaussian(key, (m, n), scale),
            # utils.generate_gaussian(key, (n,), scale),
            jnp.zeros((n,)),
        )
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]

    if len(layer_sizes) != 2:
        # for multilayer networks, remove last layer and initialize weights to 1/n
        initial_params.pop()
        hidden_dim = layer_sizes[-2]
        initial_params.append(
            (
                jnp.ones((hidden_dim, 1)) / hidden_dim,
                jnp.zeros((1,)),
            )
        )

    return initial_params


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


@partial(jax.jit, static_argnums=(2,))
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
    trial_inputs,
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
    activations = jax.vmap(network_forward, in_axes=(None, 0))(params, trial_inputs)
    # pass only the activations wrt the last odor in trial
    last_odor_activations = [a[trial_length - 1] for a in activations]
    params = update_params(
        params,
        last_odor_activations,
        plasticity_coeffs,
        plasticity_func,
        reward,
        expected_reward,
    )

    return params, (params, activations)


def update_params(
    params, activations, plasticity_coeffs, plasticity_func, reward, expected_reward
):
    """assuming plasticity happens in the first layer only.
    [dw, db] = plasticity_func(activation, reward, w, plasticity_coeffs)
    returns updated params
    """
    print("compiling model.update_params()...")
    # using expected reward or just the reward:
    reward_term = reward - expected_reward
    # reward_term = reward

    delta_params = []

    # plasticity happens in the first layer only
    activation = activations[0]
    w, b = params[0]
    # vmap over output neurons
    vmap_inputs = jax.vmap(plasticity_func, in_axes=(None, None, 0, None))
    # vmap over input neurons
    vmap_synapses = jax.vmap(vmap_inputs, in_axes=(0, None, 0, None))
    dw = vmap_synapses(activation, reward_term, w, plasticity_coeffs)
    # decide whether to update bias or not
    db = jnp.zeros_like(b)
    # db = vmap_inputs(1.0, reward_term, b, plasticity_coeffs)
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
    key, cfg, generation_coeff, generation_func, plasticity_coeff, plasticity_func, mus, sigmas
):
    """Evaluate logits, weight trajectory for generation_coeff and plasticity_coeff
       with new initial params, for a single new experiment
    Returns:
        R2 score (dict), [weights, activity]; activity is of output of plastic layer
        Percent deviance explained (scalar)
    """

    test_cfg = cfg.copy()
    # use 30% of the number of training exps for testing
    test_cfg.num_exps = (cfg.num_exps // 3) + 1
    test_cfg.trials_per_block = 80
    r2_score = {"weights": [], "activity": []}
    percent_deviance = []

    (
        xs,
        neural_recordings,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.generate_experiments_data(
        key,
        test_cfg,
        generation_coeff,
        generation_func,
        mus,
        sigmas,
    )

    for exp_i in range(test_cfg.num_exps):
        exp_i = str(exp_i)
        key, _ = jax.random.split(key)
        params = initialize_params(key, cfg, scale=0.01)
        # simulate model with "true" plasticity coefficients (generation_coeff)
        trial_lengths = jnp.sum(
            jnp.logical_not(jnp.isnan(decisions[exp_i])), axis=1
        ).astype(int)
        params_trajec, activations = simulate(
            params,
            generation_coeff,
            generation_func,
            xs[exp_i],
            rewards[exp_i],
            expected_rewards[exp_i],
            trial_lengths,
        )

        # simulate model with learned plasticity coefficients (plasticity_coeff)
        model_params_trajec, model_activations = simulate(
            params,
            plasticity_coeff,
            plasticity_func,
            xs[exp_i],
            rewards[exp_i],
            expected_rewards[exp_i],
            trial_lengths,
        )

        # simulate model with zeros plasticity coefficients for null model
        plasticity_coeff_zeros, zero_plasticity_func = synapse.init_volterra(init="zeros")
        _, null_model_activations = simulate(
            params,
            plasticity_coeff_zeros,
            zero_plasticity_func,
            xs[exp_i],
            rewards[exp_i],
            expected_rewards[exp_i],
            trial_lengths,
        )

        r2_score_exp = evaluate_r2_score(
            params_trajec, activations, model_params_trajec, model_activations
        )
        r2_score = {
            dict_key: r2_score[dict_key] + r2_score_exp[dict_key]
            for dict_key in r2_score.keys()
        }

        percent_deviance.append(
            evaluate_percent_deviance(
                decisions[exp_i], model_activations, null_model_activations
            )
        )

    return r2_score, percent_deviance


def evaluate_r2_score(
    params_trajec, activations, model_params_trajec, model_activations
):
    """
    should return a dict of R2 scores for weights and activity in
    the format of {"weights": [R2 score], "activity": [R2 score]},
    i.e. lists of length 1.
    """
    r2_score = {}

    weight_trajec = jnp.array(params_trajec[0][0])
    layer_activations = jnp.squeeze(activations[1])

    model_weight_trajec = jnp.array(model_params_trajec[0][0])
    model_layer_activations = jnp.squeeze(model_activations[1])

    r2_score["weights"] = [utils.compute_r2_score(weight_trajec, model_weight_trajec)]
    # note: this won't calculate the true R2 score between neural activity, since
    # after trial length the activities for both these would be the same (corresponding to x=0)
    r2_score["activity"] = [
        utils.compute_r2_score(layer_activations, model_layer_activations)
    ]
    return r2_score


def evaluate_percent_deviance(decisions, model_activations, null_model_activations):
    """Evaluate logits for plasticity coefficients, calculate
        neg log likelihoods. Then calculate the neg log likelihoods
        where A_ijk = 0 (null model). Then percent deviance explained
        is (D_null - D_model) / D_null
    Returns:
        Percent deviance explained scalar
    """

    trial_lengths = jnp.sum(jnp.logical_not(jnp.isnan(decisions)), axis=1).astype(int)
    logits_mask = np.ones_like(decisions)
    for j, length in enumerate(trial_lengths):
        logits_mask[j][length:] = 0
    decisions = jnp.nan_to_num(decisions, copy=False, nan=0.0)

    ys = jax.nn.sigmoid(jnp.squeeze(model_activations[-1]))
    ys = jnp.multiply(ys, logits_mask)
    null_ys = jax.nn.sigmoid(jnp.squeeze(null_model_activations[-1]))
    null_ys = jnp.multiply(null_ys, logits_mask)
    # print("ys:", ys)
    # print("null ys:", null_ys)

    model_deviance = utils.compute_neg_log_likelihoods(logits_mask, ys, decisions)
    null_deviance = utils.compute_neg_log_likelihoods(logits_mask, null_ys, decisions)
    print(f"model deviance: {model_deviance}")
    print(f"null deviance: {null_deviance}")
    percent_deviance = 100 * (null_deviance - model_deviance) / null_deviance
    return percent_deviance.item()
