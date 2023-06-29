import jax
import jax.numpy as jnp
from functools import partial
import optax
import plasticity.behavior.model as model


def compute_cross_entropy(decisions, logits):
    # returns the mean of the element-wise cross entropy
    losses = optax.sigmoid_binary_cross_entropy(logits, decisions)
    return jnp.mean(losses)


@partial(jax.jit, static_argnames=["plasticity_func"])
def celoss(
    winit,
    plasticity_coeff,
    plasticity_func,
    xs,
    rewards,
    expected_rewards,
    decisions,
    trial_lengths,
    logits_mask,
    coeff_mask,
):

    plasticity_coeff = jnp.multiply(plasticity_coeff, coeff_mask)

    logits, _ = model.simulate(
        winit,
        plasticity_coeff,
        plasticity_func,
        xs,
        rewards,
        expected_rewards,
        trial_lengths,
    )
    # print("decisions: \n", decisions)
    logits = jnp.multiply(logits, logits_mask)
    decisions = jnp.nan_to_num(decisions, copy=False, nan=0)
    # print("logits: \n", logits)

    loss = compute_cross_entropy(decisions, logits)

    # add a L1 regularization term to the loss
    # loss += 5e-5 * jnp.sum(jnp.abs(plasticity_coeff))
    return loss
