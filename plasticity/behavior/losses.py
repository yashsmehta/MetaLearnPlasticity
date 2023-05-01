import jax
import jax.numpy as jnp
from functools import partial
import optax
import plasticity.behavior.network as network


def compute_cross_entropy(fly_ys, insilico_ys):
    losses = [
        jnp.mean(optax.sigmoid_binary_cross_entropy(jnp.array(fly_trial), jnp.array(insilico_trial)))
        for fly_block, insilico_block in zip(fly_ys, insilico_ys)
        for fly_trial, insilico_trial in zip(fly_block, insilico_block)
    ]
    loss = jnp.mean(jnp.array(losses))
    return loss


# @partial(jax.jit, static_argnames=["plasticity_func"])
def celoss(
        winit,
        plasticity_coeff,
        plasticity_func,
        xs,
        rewards,
        exp_rewards,
        fly_ys):

    insilico_ys = network.simulate_insilico_experiment(winit, plasticity_coeff, plasticity_func, xs, rewards, exp_rewards)
    loss = compute_cross_entropy(fly_ys, insilico_ys)

    # add a L1 regularization term to the loss
    # loss += 5e-6 * jnp.sum(jnp.abs(student_coefficients))
    return loss
