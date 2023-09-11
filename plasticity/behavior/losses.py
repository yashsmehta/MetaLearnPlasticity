import jax
import jax.numpy as jnp
from functools import partial
import optax
import plasticity.behavior.model as model
import numpy as np


def compute_cross_entropy(decisions, logits):
    # returns the mean of the element-wise cross entropy
    losses = optax.sigmoid_binary_cross_entropy(logits, decisions)
    return jnp.mean(losses)


def compute_mse(neural_recordings, layer_activations):
    # returns the mean of the element-wise mse
    losses = optax.squared_error(neural_recordings, layer_activations)
    return jnp.mean(losses)


def compute_neural_loss(logits_mask, recording_sparsity, neural_recordings, layer_activations):
    # sparsify the neural recordings, and then compute the mse
    recording_sparsity = float(recording_sparsity)
    num_neurons = neural_recordings.shape[-1]
    num_recorded_neurons = int(recording_sparsity * num_neurons)

    recordings_id = np.random.choice(
        np.arange(num_neurons), size=num_recorded_neurons, replace=False
    )
    recordings_mask = np.zeros(num_neurons)
    recordings_mask[recordings_id] = 1.
    neural_recordings = jnp.einsum('ijk, k -> ijk', neural_recordings, recordings_mask)
    layer_activations = jnp.einsum('ijk, k -> ijk', layer_activations, recordings_mask)
    # layer activations need to be masked as well for trials that are shorter than the max trial length,
    # since zeros are fed as inputs, after sigmpoid, the activations will be 0.5
    layer_activations = jnp.einsum('ijk, ij -> ijk', layer_activations, logits_mask)
    neural_loss = compute_mse(neural_recordings, layer_activations)
    return neural_loss


@partial(jax.jit, static_argnames=["plasticity_func", "cfg"])
def celoss(
    params,
    plasticity_coeff,
    plasticity_func,
    xs,
    rewards,
    expected_rewards,
    neural_recordings,
    decisions,
    logits_mask,
    cfg,
):
    coeff_mask = jnp.array(cfg.coeff_mask)
    plasticity_coeff = jnp.einsum('ijkl, jkl -> ijkl', plasticity_coeff, coeff_mask)
    trial_lengths = jnp.sum(logits_mask, axis=1).astype(int)

    params_trajec, activations = model.simulate(
        params,
        plasticity_coeff,
        plasticity_func,
        xs,
        rewards,
        expected_rewards,
        trial_lengths,
    )
    # why does logit_mask have an effect??!!
    logits = jnp.squeeze(activations[-1])
    # mask out the logits after the trial length
    logits = jnp.multiply(logits, logits_mask)
    decisions = jnp.nan_to_num(decisions, copy=False, nan=0.0)

    behavior_loss = compute_cross_entropy(decisions, logits)
    neural_loss = compute_neural_loss(logits_mask, cfg.neural_recording_sparsity, neural_recordings, activations[-2])
    # add a L1 regularization term to the loss
    loss = cfg.l1_regularization * jnp.sum(jnp.abs(plasticity_coeff))
    # python string search if "neural" in cfg.fit_data
    if "neural" in cfg.fit_data:
        loss += neural_loss
    if "behavior" in cfg.fit_data:
        loss += behavior_loss

    return loss
