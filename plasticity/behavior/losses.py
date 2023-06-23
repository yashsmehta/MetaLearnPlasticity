import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import optax
from jax.experimental.host_callback import id_print
import plasticity.behavior.network as network


def compute_cross_entropy(decisions, outputs):
    losses = optax.sigmoid_binary_cross_entropy(outputs, decisions)
    losses = jnp.nan_to_num(losses, nan=0)
    return jnp.sum(losses)


# @partial(jax.jit, static_argnames=["plasticity_func"])
def celoss(
        winit,
        plasticity_coeff,
        plasticity_func,
        xs,
        rewards,
        exp_rewards,
        decisions,
        trial_lengths,
        plasticity_mask=np.ones((3, 3, 3))
        ):

    plasticity_coeff = jnp.multiply(plasticity_coeff, plasticity_mask)

    # here the outputs are not changing! Debug this!
    outputs, _ = network.simulate_insilico_experiment(
        winit, plasticity_coeff, plasticity_func, xs, rewards, exp_rewards, trial_lengths
    )
    # id_print(outputs)

    loss = compute_cross_entropy(decisions, outputs)

    # add a L1 regularization term to the loss
    # loss += 5e-6 * jnp.sum(jnp.abs(student_coefficients))
    return loss
