import plasticity.inputs as inputs
from plasticity import synapse
import plasticity.data_loader as data_loader
import plasticity.losses as losses
import plasticity.model as model
import plasticity.utils as utils
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax.random import split
import pandas as pd
from statistics import mean
import time
import pickle
import sys


def train(cfg):
    jax.config.update("jax_platform_name", "cpu")
    utils.assert_valid_config(cfg)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    key = jax.random.PRNGKey(cfg.jobid)
    generation_coeff, generation_func = synapse.init_plasticity(
        key, cfg, mode="generation_model"
    )
    key, subkey = split(key)
    plasticity_coeff, plasticity_func = synapse.init_plasticity(
        subkey, cfg, mode="plasticity_model"
    )

    params = model.initialize_params(key, cfg)
    mus, sigmas = inputs.generate_input_parameters(cfg)

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print(f"layer size: {cfg.layer_sizes}")

    key, _ = split(key)

    if cfg.use_experimental_data:
        (
            resampled_xs,
            decisions,
            rewards,
            expected_rewards,
        ) = data_loader.load_single_adi_experiment(cfg)
    else:
        (
            resampled_xs,
            neural_recordings,
            decisions,
            rewards,
            expected_rewards,
        ) = data_loader.generate_experiments_data(
            key,
            cfg,
            generation_coeff,
            generation_func,
            mus,
            sigmas,
        )
    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=2)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(plasticity_coeff)
    expdata = {}

    for epoch in range(cfg.num_epochs):
        for exp_i in range(cfg.num_exps):
            noise_key = jax.random.PRNGKey((cfg.jobid + 1) * (exp_i + 1))
            exp_i = str(exp_i)

            # loss = losses.celoss(
            #     key,
            #     params,
            #     plasticity_coeff,
            #     plasticity_func,
            #     resampled_xs[str(exp_i)],
            #     rewards[str(exp_i)],
            #     expected_rewards[str(exp_i)],
            #     neural_recordings[str(exp_i)],
            #     decisions[str(exp_i)],
            #     cfg,
            # )
            # print(f"loss: {loss}")
            # exit()

            loss, meta_grads = loss_value_and_grad(
                noise_key,
                params,
                plasticity_coeff,
                plasticity_func,
                resampled_xs[exp_i],
                rewards[exp_i],
                expected_rewards[exp_i],
                neural_recordings[exp_i],
                decisions[exp_i],
                cfg,
            )

            updates, opt_state = optimizer.update(
                meta_grads, opt_state, plasticity_coeff
            )

            plasticity_coeff = optax.apply_updates(plasticity_coeff, updates)

        if epoch % cfg.log_interval == 0:
            expdata = utils.print_and_log_training_info(
                cfg, expdata, plasticity_coeff, epoch, loss
            )

    key, _ = split(key)
    r2_score, percent_deviance = model.evaluate(
        key,
        cfg,
        generation_coeff,
        generation_func,
        plasticity_coeff,
        plasticity_func,
        mus,
        sigmas,
    )

    print(f"r2 score: {r2_score}")
    print(f"percent deviance: {percent_deviance}")
    if cfg.plasticity_model == "mlp":
        mlp_params = expdata.pop("mlp_params")

    df = pd.DataFrame.from_dict(expdata)
    df["r2_weights"], df["r2_activity"] = mean(r2_score["weights"]), mean(
        r2_score["activity"]
    )
    df["percent_deviance"] = mean(percent_deviance)

    for key, value in cfg.items():
        if isinstance(value, (float, int)):
            df[key] = value

    # pd.set_option("display.max_columns", None)
    print(df.tail(5))
    logdata_path = utils.save_logs(cfg, df)
    if cfg.plasticity_model == "mlp" and cfg.log_expdata:
        with open(logdata_path / f"mlp_params_{cfg.jobid}.pkl", "wb") as f:
            pickle.dump(mlp_params, f)
