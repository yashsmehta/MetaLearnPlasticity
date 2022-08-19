import time
import jax
from jax.tree_util import Partial
import jax.numpy as jnp

import plastix as px
import dataloaders.toy_ds as ds


def mse_loss(layer, y, state, parameters):
    state = layer.update_state(state, parameters)
    return jnp.mean(jnp.square(y - state.output_nodes.rate))


def main():
    dataset = ds.AndDataSet()
    key = jax.random.PRNGKey(round(time.time()))
    x, y = dataset.get_noisy_samples(num=1, key=key, sigma=0.0)
    print("x:", x)
    print("y:", y)
    lr = 0.2
    edges = [(0, 0), (1, 0)]

    layer = px.layers.SparseLayer(
        2, 1, edges, px.kernels.edges.FixedWeight(), px.kernels.nodes.SumNonlinear()
    )
    state = layer.init_state()
    parameters = layer.init_parameters()

    state.input_nodes.rate = x
    state = layer.update_state(state, parameters)
    # state = layer.update_state(state, parameters)
    loss = Partial((mse_loss), layer)
    grads = jax.grad(loss, argnums=2)(y, state, parameters)
    parameters.edges.weight += lr * grads.edges.weight
    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
