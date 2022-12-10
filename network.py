import jax
import jax.numpy as jnp
from jax.lax import reshape
from jax import vmap

non_linear = True

def generate_trajectory(
        input_sequence,
        initial_weights,
        volterra_coefficients):
    """
    generate a single trajectory given an input sequence, initial weights
    and the "meta" plasticity coefficients
    """

    def step(weights, inputs):
        return volterra_update_weights(inputs, weights, volterra_coefficients)

    final_weights, activity_trajec = jax.lax.scan(
        step, initial_weights, input_sequence
        )
    return activity_trajec

def forward(inputs, weights):
    activatation = inputs @ weights
    if non_linear:
        activatation = jax.nn.sigmoid(activatation)
    return activatation

def volterra_update_weights(inputs, weights, volterra_coefficients):
    activation = forward(inputs, weights)
    m, n = weights.shape
    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(activation, (n,)), indexing="ij"
    )

    dw = vmap(volterra_synaptic_dw, in_axes=(0, 0, 0, None))(
            reshape(in_grid, (m * n, 1)),
            reshape(out_grid, (m * n, 1)),
            reshape(weights, (m * n, 1)),
            volterra_coefficients)
    
    dw = reshape(dw, (m, n))
    assert (
        dw.shape == weights.shape
    ), "dw and w should be of the same shape to prevent broadcasting while adding"
    weights += dw

    return (weights, activation)


def volterra_synaptic_dw(pre, post, weight, volterra_coefficients):
    synapse_tensor = get_synapse_tensor(pre, post, weight)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw


def get_synapse_tensor(pre, post, weight):
    synapse_tensor = jnp.outer(
        jnp.outer(
            jnp.array([pre**0, pre**1, pre**2]),
            jnp.array([post**0, post**1, post**2])),
        jnp.array([weight**0, weight**1, weight**2]))
    
    synapse_tensor = jnp.reshape(synapse_tensor, (3, 3, 3))
    return synapse_tensor

def mlp_update_weights(self, inputs, weights, plasticity_mlp):
    act = self.forward(inputs, weights)
    n, m = weights.shape
    in_grid, out_grid = jnp.meshgrid(
        reshape(inputs, (m,)), reshape(act, (n,)), indexing="ij"
    )

    local_info = jnp.hstack(
        (
            reshape(weights, (m * n, 1)),
            reshape(in_grid, (m * n, 1)),
            reshape(out_grid, (m * n, 1)),
        )
    )

    dw = vmap(mlp_synaptic_dw, in_axes=(0, None))(local_info, plasticity_mlp)
    dw = reshape(dw, (n, m))

    assert (
        dw.shape == weights.shape
    ), "dw and w should be of the same shape to prevent broadcasting while adding"
    weights += dw

    return (weights, act)


def mlp_synaptic_dw(local_info, plasticity_mlp):
    # inputs shape: (3,)
    act = local_info
    for params in plasticity_mlp:
        h = jnp.dot(params, act)
        act = jax.nn.tanh(h)

    return act
