import jax
from jax.tree_util import Partial
import jax.numpy as jnp

import optax
import plastix as px
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import dataloaders.toy_ds as ds


def mse_loss(layer, y, state, parameters):
    state = layer.update_state(state, parameters)
    return jnp.mean(optax.l2_loss(state.output_nodes.rate, y))


def main():
    dataset = ds.AndDataSet()
    key = jax.random.PRNGKey(0)
    x, y = dataset.get_noisy_samples(num=1, key=key, sigma=0.0)
    max_iter = 20
    edges = [(0, 0), (1, 0)]

    layer = px.layers.SparseLayer(
        2,
        1,
        edges,
        px.kernels.edges.FixedWeight(),
        px.kernels.nodes.SumNonlinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()

    state = layer.update_state(state, parameters)
    loss = Partial((mse_loss), layer)
    optimizer = optax.rmsprop(learning_rate=1e-3)
    opt_state = optimizer.init(parameters.edges.weight)

    for _ in range(max_iter):
        key, _ = jax.random.split(key)
        x, y = dataset.get_noisy_samples(num=1, key=key, sigma=0.0)
        state.input_nodes.rate = x
        grads = jax.grad(loss, argnums=2)(y, state, parameters)
        updates, opt_state = optimizer.update(grads.edges.weight, opt_state, parameters.edges.weight)
        parameters.edges.weight = optax.apply_updates(parameters.edges.weight, updates)

    state = layer.update_state(state, parameters)
    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
