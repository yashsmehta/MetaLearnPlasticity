import jax.numpy as jnp
import numpy as np
import jax


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


def truncated_sigmoid(x):
    epsilon = 1e-6
    return jnp.clip(jax.nn.sigmoid(x), epsilon, 1 - epsilon)
