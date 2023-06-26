import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import optax
from jax.experimental.host_callback import id_print
import plasticity.behavior.network as network


def compute_cross_entropy(decisions, logits):
    losses = optax.sigmoid_binary_cross_entropy(logits, decisions)
    return jnp.mean(losses)


@partial(jax.jit, static_argnames=["plasticity_func"])
def celoss(
        winit,
        plasticity_coeff,
        plasticity_func,
        xs,
        rewards,
        exp_rewards,
        decisions,
        trial_lengths,
        mask,
        plasticity_mask=np.ones((3, 3, 3))
        ):

    plasticity_coeff = jnp.multiply(plasticity_coeff, plasticity_mask)

    logits, _ = network.simulate_insilico_experiment(
        winit, plasticity_coeff, plasticity_func, xs, rewards, exp_rewards, trial_lengths
    )
    # print("decisions: \n", decisions)
    logits = jnp.multiply(logits, mask)
    decisions = jnp.nan_to_num(decisions, copy=False, nan=0)
    # print("logits: \n", logits)
 
    loss = compute_cross_entropy(decisions, logits)

    # add a L1 regularization term to the loss
    # loss += 5e-6 * jnp.sum(jnp.abs(student_coefficients))
    return loss
