import jax.numpy as jnp
import jax
from jax import vmap
from jax.lax import reshape


def simulate_insilico_experiment(
    initial_weights,
    plasticity_coeffs,
    plasticity_func,
    xs,
    rewards,
    expected_rewards,
    trial_lengths,
):
    """Simulate an experiment with given plasticity coefficients,
       vmap over timesteps within a trial, and scan over all trials
    Returns:
        a tensor of logits for the experiment, and the final weights
    """
    def step(weights, stimulus):
        x, reward, expected_reward, trial_length = stimulus
        return network_step(
            x,
            weights,
            plasticity_coeffs,
            plasticity_func,
            reward,
            expected_reward,
            trial_length,
        )

    final_weights, logits = jax.lax.scan(
        step, initial_weights, (xs, rewards, expected_rewards, trial_lengths)
    )
    return jnp.squeeze(logits), final_weights


def network_step(
    input,
    weights,
    plasticity_coeffs,
    plasticity_func,
    reward,
    expected_reward,
    trial_length,
):
    """Performs a forward pass and weight update
        Forward pass is needed to compute logits for the loss function
    Returns:
        updated weights and logits
    """
    vmapped_forward = vmap(lambda weights, x: jnp.dot(x, weights), (None, 0))
    logits = vmapped_forward(weights, input)
    x = input[trial_length - 1]
    dw = weight_update(
        x, weights, plasticity_coeffs, plasticity_func, reward, expected_reward
    )
    weights += dw

    return weights, logits


def weight_update(
    x, weights, plasticity_coeffs, plasticity_func, reward, expected_reward
):
    """Weight update for all synapses in the layer. Uses vmap to vectorize
       plasticity function over all synapses
    Returns:
        delta weights for the layer
    """
    reward_term = reward - expected_reward
    m, n = weights.shape
    in_grid, _ = jnp.meshgrid(
        reshape(x, (m,)),
        jnp.ones(
            n,
        ),
        indexing="ij",
    )

    vfun = vmap(plasticity_func, in_axes=(0, None, 0, None))
    dw = vmap(vfun, in_axes=(1, None, 1, None), out_axes=1)(
        in_grid, reward_term, weights, plasticity_coeffs
    )

    assert (
        dw.shape == weights.shape
    ), "dw and w should be of the same shape to prevent broadcasting while adding"

    return dw
