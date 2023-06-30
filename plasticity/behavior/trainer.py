from plasticity.behavior import utils
import plasticity.inputs as inputs
from plasticity import synapse
from plasticity.behavior.utils import coeff_logs_to_dict
import plasticity.behavior.data_loader as data_loader
import plasticity.behavior.losses as losses
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax.random import split
from pathlib import Path
import pandas as pd
import time


def train(cfg):
    coeff_mask = np.array(cfg.coeff_mask)
    key = jax.random.PRNGKey(cfg.seed)
    simulation_coeff, plasticity_func = synapse.init_reward_volterra(init="reward")
    plasticity_coeff, _ = synapse.init_reward_volterra(init="zeros")

    key, _ = split(key)

    # winit = jnp.zeros((input_dim, output_dim))
    winit = utils.generate_gaussian(key, (cfg.input_dim, cfg.output_dim), scale=0.01)
    print(f"initial weights: \n{winit}")
    key, _ = split(key)

    mus, sigmas = inputs.generate_binary_input_parameters()

    # are we running on CPU or GPU?
    device = jax.lib.xla_bridge.get_backend().platform
    print("platform: ", device)
    print("layer size: [{}, {}]".format(cfg.input_dim, cfg.output_dim))
    print()

    (
        xs,
        odors,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.generate_experiments_data(
        key,
        cfg,
        winit,
        simulation_coeff,
        plasticity_func,
        mus,
        sigmas,
    )


    loss_value_and_grad = jax.value_and_grad(losses.celoss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(plasticity_coeff)
    coeff_logs, epoch_logs = [], []

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

            loss, meta_grads = loss_value_and_grad(
                winit,
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
            print(
                plasticity_coeff[1, 1, 0],
                plasticity_coeff[1, 0, 0],
                plasticity_coeff[0, 1, 0],
                plasticity_coeff[0, 0, 0],
            )
            print()
            coeff_logs.append(plasticity_coeff)
            epoch_logs.append(epoch)

    coeff_logs = np.array(coeff_logs)
    expdata = coeff_logs_to_dict(coeff_logs, coeff_mask)
    expdata["epoch"] = epoch_logs
    df = pd.DataFrame.from_dict(expdata)

    for key, value in cfg.items():
        if(isinstance(value, (float, int))):
            df[key] = value
            print(key, value)
    pd.set_option("display.max_columns", None)
    print(df.tail(5))

    if cfg.log_expdata:
        logdata_path = Path(cfg.log_dir) / f"{cfg.exp_name}"
        logdata_path.mkdir(parents=True, exist_ok=True)

        csv_file = logdata_path / "logs.csv"
        write_header = not csv_file.exists()
        df.to_csv(csv_file, mode="a", header=write_header)
