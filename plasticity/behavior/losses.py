import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import optax
from jax.experimental.host_callback import id_print
import plasticity.behavior.network as network


def compute_cross_entropy(fly_ys, insilico_ys):
    losses = [
        jnp.mean(optax.sigmoid_binary_cross_entropy(jnp.array(fly_trial), jnp.array(insilico_trial)))
        for fly_block, insilico_block in zip(fly_ys, insilico_ys)
        for fly_trial, insilico_trial in zip(fly_block, insilico_block)
    ]
    loss = jnp.mean(jnp.array(losses))
    return loss


@partial(jax.jit, static_argnames=["plasticity_func"])
def celoss(
        winit,
        plasticity_coeff,
        plasticity_func,
        xs,
        rewards,
        exp_rewards,
        fly_ys,
        plasticity_mask=np.ones((3, 3, 3))
        ):

    plasticity_coeff = jnp.multiply(plasticity_coeff, plasticity_mask)
    insilico_ys = network.simulate_insilico_experiment(winit, plasticity_coeff, plasticity_func, xs, rewards, exp_rewards)
    id_print(plasticity_coeff[1][1][0])
    loss = compute_cross_entropy(fly_ys, insilico_ys)

    # add a L1 regularization term to the loss
    # loss += 5e-6 * jnp.sum(jnp.abs(student_coefficients))
    return loss
