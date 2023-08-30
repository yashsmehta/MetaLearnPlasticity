from plasticity.behavior import utils
import plasticity.inputs as inputs
from plasticity import synapse
from plasticity.behavior.utils import coeff_logs_to_dict
import plasticity.behavior.data_loader as data_loader
import plasticity.behavior.losses as losses
import plasticity.behavior.model as model
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax.random import split
from pathlib import Path
import pandas as pd
import time


def train(cfg):
    jax.config.update('jax_platform_name', 'cpu')
    coeff_mask = np.array(cfg.coeff_mask)
    key = jax.random.PRNGKey(cfg.jobid)
    np.random.seed(cfg.jobid)
    generation_coeff, plasticity_func = synapse.init_volterra(init="reward")
    plasticity_coeff, _ = synapse.init_volterra(key, init="reward")

    key, key2 = split(key)
    params = model.initialize_params(key2, cfg)
    mus, sigmas = inputs.generate_input_parameters(cfg)

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print(f"layer size: {cfg.layer_sizes}")
    start = time.time()

    if cfg.use_experimental_data:
        (
            xs,
            decisions,
            rewards,
            expected_rewards,
        ) = data_loader.load_single_adi_experiment(cfg)
    else:
        (
            xs,
            odors,
            decisions,
            rewards,
            expected_rewards,
        ) = data_loader.generate_experiments_data(
            key,
            cfg,
            params,
            generation_coeff,
            plasticity_func,
            mus,
            sigmas,
        )

    print(f"got training data in {time.time() - start} seconds!")

    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(plasticity_coeff)
    coeff_logs, epoch_logs, loss_logs = [], [], []

    for epoch in range(cfg.num_epochs):
        for exp_i in range(cfg.num_exps):
            start = time.time()
            # calculate the length of each trial by checking for NaNs
            trial_lengths = jnp.sum(
                jnp.logical_not(jnp.isnan(decisions[str(exp_i)])), axis=1
            ).astype(int)

            logits_mask = np.ones(decisions[str(exp_i)].shape)
            for j, length in enumerate(trial_lengths):
                logits_mask[j][length:] = 0

            # loss = losses.celoss(
            #     params,
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

            loss, meta_grads = loss_value_and_grad(
                params,
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
        if epoch % cfg.log_interval == 0:
            print(f"epoch :{epoch}")
            print(f"loss :{loss}")
            indices = coeff_mask.nonzero()
            ind_i, ind_j, ind_k = indices
            for index in zip(ind_i, ind_j, ind_k):
                print(f"A{index}", plasticity_coeff[index])
            print("\n")
            coeff_logs.append(plasticity_coeff)
            epoch_logs.append(epoch)
            loss_logs.append(loss)

    (logits, weight_trajec), (model_logits, model_weight_trajec) = model.evaluate(
        key,
        cfg,
        generation_coeff,
        plasticity_coeff,
        plasticity_func,
        mus,
        sigmas,
    )

    r2_score = utils.r2_score(weight_trajec, model_weight_trajec)
    kl_div = utils.kl_divergence(logits, model_logits)
    print(f"r2 score: {r2_score}")
    print(f"kl divergence: {kl_div}")

    coeff_logs = np.array(coeff_logs)
    expdata = coeff_logs_to_dict(coeff_logs)
    expdata["epoch"] = epoch_logs
    expdata["loss"] = loss_logs
    df = pd.DataFrame.from_dict(expdata)
    df["r2_score"] = r2_score

    for key, value in cfg.items():
        if isinstance(value, (float, int)):
            df[key] = value
            # print(key, value)

    pd.set_option("display.max_columns", None)
    print(df.tail(5))

    if cfg.log_expdata:
        logdata_path = Path(cfg.log_dir)
        if cfg.use_experimental_data:
            logdata_path = logdata_path / "expdata" / cfg.exp_name
        else:
            logdata_path = logdata_path / "simdata" / cfg.exp_name

        logdata_path.mkdir(parents=True, exist_ok=True)
        csv_file = logdata_path / f"exp_{cfg.jobid}.csv"
        write_header = not csv_file.exists()
        df.to_csv(csv_file, mode="a", header=write_header, index=False)
