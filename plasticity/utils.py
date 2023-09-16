import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from scipy.special import kl_div
from pathlib import Path
import os


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def compute_neg_log_likelihoods(ys, decisions):
    not_ys = jnp.ones_like(ys) - ys
    neg_log_likelihoods = -2 * jnp.log(jnp.where(decisions == 1, ys, not_ys))
    return jnp.mean(neg_log_likelihoods)

def kl_divergence(logits1, logits2):
    """
    computes the KL divergence between two probability distributions,
    p and q would be logits
    """
    p = jax.nn.softmax(logits1)
    q = jax.nn.softmax(logits2)
    kl_matrix = kl_div(p, q)
    return np.sum(kl_matrix)

def create_nested_list(num_outer, num_inner):
    return [[[] for _ in range(num_inner)] for _ in range(num_outer)]

def truncated_sigmoid(x, epsilon=1e-6):
    return jnp.clip(jax.nn.sigmoid(x), epsilon, 1 - epsilon)

def experiment_list_to_tensor(longest_trial_length, nested_list, list_type):
    # note: trial lengths are not all the same, so we need to pad with nans
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])

    num_trials = num_blocks * trials_per_block

    if list_type == "decisions" or list_type == "odors":
        # individual trial length check for nans and stop when they see one
        tensor = np.full((num_trials, longest_trial_length), np.nan)

    elif list_type == "xs" or list_type == "neural_recordings":
        element_dim = len(nested_list[0][0][0])
        tensor = np.full((num_trials, longest_trial_length, element_dim), 0.)
    else:
        raise Exception("list passed must be odors, decisions or xs")

    for i in range(num_blocks):
        for j in range(trials_per_block):
            trial = nested_list[i][j]
            for k in range(len(trial)):
                tensor[i * trials_per_block + j][k] = trial[k]

    return jnp.array(tensor)


def print_and_log_training_info(cfg, expdata, plasticity_coeff, epoch, loss):

    print(f"epoch :{epoch}")
    print(f"loss :{loss}")

    if cfg.plasticity_model == "volterra":
        coeff_mask = np.array(cfg.coeff_mask)
        plasticity_coeff = np.multiply(plasticity_coeff, coeff_mask)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dict_key = f"A_{i}{j}{k}"
                    expdata.setdefault(dict_key, []).append(plasticity_coeff[i, j, k])

        ind_i, ind_j, ind_k = coeff_mask.nonzero()
        for index in zip(ind_i, ind_j, ind_k):
            print(f"A{index}", plasticity_coeff[index])
        print("\n")
    else:
        print("MLP plasticity coeffs: ", plasticity_coeff)
        expdata.setdefault("mlp_params", []).append(plasticity_coeff)

    expdata.setdefault("epoch", []).append(epoch)
    # check if loss is nan
    if np.isnan(loss):
        print("loss is nan!, exiting...")
        exit()
    expdata.setdefault("loss", []).append(loss)

    return expdata


def save_logs(cfg, df):
    logdata_path = Path(cfg.log_dir)
    if cfg.log_expdata:
        if cfg.use_experimental_data:
            logdata_path = logdata_path / "expdata" / cfg.exp_name
        else:
            logdata_path = logdata_path / "simdata" / cfg.exp_name

        logdata_path.mkdir(parents=True, exist_ok=True)
        csv_file = logdata_path / f"exp_{cfg.jobid}.csv"
        write_header = not csv_file.exists()
        df.to_csv(csv_file, mode="a", header=write_header, index=False)
        print("saved logs!")
    return logdata_path


def assert_valid_config(cfg):
    """
    asserts that the configuration is valid
    """
    assert (
        len(cfg.reward_ratios) == cfg.num_blocks
    ), "length of reward ratios should be equal to number of blocks!"
    assert cfg.plasticity_model in ["volterra", "mlp"], "only volterra, mlp plasticity model supported!"
    assert cfg.generation_model in ["volterra", "mlp"], "only volterra, mlp generation model supported!"
    assert cfg.meta_mlp_layer_sizes[0] == 3 and cfg.meta_mlp_layer_sizes[-1] == 1, "meta mlp input dim must be 3, and output dim 1!"
    assert cfg.layer_sizes[-1] == 1, "output dim must be 1!"
    assert len(cfg.layer_sizes) == 2 or len(cfg.layer_sizes) == 3, "only 2, 3 layer networks supported!"
    assert cfg.neural_recording_sparsity >= 0. and cfg.neural_recording_sparsity <= 1., "neural recording sparsity must be between 0 and 1!"
    if cfg.use_experimental_data:
        assert "behavior" in cfg.fit_data and "neural" not in cfg.fit_data, "only behavior experimental data available!"
    if cfg.plasticity_model == "mlp":
        assert cfg.plasticity_coeff_init in ["random"], "only random plasticity coeff init for MLP supported!"
    assert "behavior" in cfg.fit_data or "neural" in cfg.fit_data, "fit data must contain either behavior or neural, or both!"
    if cfg.use_experimental_data:
        assert cfg.num_exps <= len(os.listdir(cfg.data_dir)), "Not enough experimental data"
    return