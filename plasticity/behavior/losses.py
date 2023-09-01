import jax
import jax.numpy as jnp
from functools import partial
import optax
import plasticity.behavior.model as model


def compute_cross_entropy(decisions, logits):
    # returns the mean of the element-wise cross entropy
    losses = optax.sigmoid_binary_cross_entropy(logits, decisions)
    return jnp.mean(losses)

def compute_mse(neural_recordings, layer_activations):
    # returns the mean of the element-wise mse
    losses = optax.squared_error(neural_recordings, layer_activations)
    return jnp.mean(losses)

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
    plasticity_coeff = jnp.multiply(plasticity_coeff, coeff_mask)
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
    decisions = jnp.nan_to_num(decisions, copy=False, nan=0.)

    ce_loss = compute_cross_entropy(decisions, logits)
    mse_loss = compute_mse(neural_recordings, activations[-2])
    # add a L1 regularization term to the loss
    loss = cfg.l1_regularization * jnp.sum(jnp.abs(plasticity_coeff))
    # python string search if "neural" in cfg.fit_data
    if "neural" in cfg.fit_data:
        loss += mse_loss
    if "behavior" in cfg.fit_data:
        loss += ce_loss

    return loss
