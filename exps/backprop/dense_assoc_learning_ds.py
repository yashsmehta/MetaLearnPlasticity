import jax
from jax.tree_util import Partial
import jax.numpy as jnp

import sys
import plastix as px

import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import  dataloaders.associative_learning_ds as ds
 

def mse_loss(layer, y, state, parameters):
    state = layer.update_state(state, parameters)
    return jnp.mean(jnp.square(y - state.output_nodes.rate))


def main():
    key = jax.random.PRNGKey(0)
    s, u, rtarg, trialinfo = ds.genrandtrials()
    print(s, u, rtarg, trialinfo) 
    lr = 0.2
    max_iter = 5

    layer = px.layers.DenseLayer(
        2,
        1,
        px.kernels.edges.FixedWeight(),
        px.kernels.nodes.SumNonlinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()

    state = layer.update_state(state, parameters)
    loss = Partial((mse_loss), layer)

    for _ in range(max_iter):
        key, _ = jax.random.split(key)
        x, y = dataset.get_noisy_samples(num=1, key=key, sigma=0.0)
        state.input_nodes.rate = x
        grads = jax.grad(loss, argnums=2)(y, state, parameters)
        parameters.edges.weight += lr * grads.edges.weight

    state = layer.update_state(state, parameters)
    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
