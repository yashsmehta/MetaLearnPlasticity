import jax
import jax.numpy as jnp
import numpy as np
import sklearn.metrics
from scipy.special import kl_div


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def generate_random_connectivity(key, m, n, sparsity):
    """
    returns a random binary connectivity mask of shape [M x N]
    with a specific connectivity sparseness
    """


def r2_score(tensor1, tensor2):
    tensor1 = tensor1.reshape(tensor1.shape[0], -1)
    tensor2 = tensor2.reshape(tensor2.shape[0], -1)
    return sklearn.metrics.r2_score(tensor1, tensor2)


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


def experiment_list_to_tensor(longest_trial_length, nested_list, list_type):
    # note: trial lengths are not all the same, so we need to pad with nans
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])

    num_trials = num_blocks * trials_per_block

    if list_type == "decisions" or list_type == "odors":
        # individual trial length check for nans and stop when they see one
        tensor = np.full((num_trials, longest_trial_length), np.nan)

    elif list_type == "xs":
        element_dim = len(nested_list[0][0][0])
        tensor = np.full((num_trials, longest_trial_length, element_dim), 0)
    else:
        raise Exception("list passed must be odors, decisions or xs")

    for i in range(num_blocks):
        for j in range(trials_per_block):
            trial = nested_list[i][j]
            for k in range(len(trial)):
                tensor[i * trials_per_block + j][k] = trial[k]

    return jnp.array(tensor)


def coeff_logs_to_dict(coeff_logs, coeff_mask):
    """Converts coeff_logs to a dictionary with keys corresponding to
    the plasticity coefficient names (A_ijk), for i,j,k where coeff_mask[i,j,k] = 1
    """
    coeff_dict = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if coeff_mask[i, j, k] == 1:
                    coeff_dict[f"A_{i}{j}{k}"] = coeff_logs[:, i, j, k]

    return coeff_dict


def truncated_sigmoid(x, epsilon=1e-6):
    return jnp.clip(jax.nn.sigmoid(x), epsilon, 1 - epsilon)
