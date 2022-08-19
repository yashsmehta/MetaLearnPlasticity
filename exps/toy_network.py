import numpy as np
import plastix as px
import jax
import jax.numpy as jnp
from dataloaders.toy_ds import AndDataSet


def get_reward(y, state):
    pred = state.output_nodes.rate
    if round(pred) == y:
        r = 1
    else:
        r = 0
    return r


def main():
    dataset = AndDataSet()
    key = jax.random.PRNGKey(0)
    x, y = dataset.get_noisy_samples(1, key)
    edges = [(0, 0), (1, 1)]

    layer = px.layers.SparseLayer(
        2,
        2,
        edges,
        px.kernels.edges.RatePolynomialUpdate(),
        px.kernels.nodes.SumNonlinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()

    state.input_nodes.rate = x
    parameters.edges.weight *= 0.5
    parameters.edges.lr = jnp.array(0.1)
    # parameters.edges.A['111'] = jnp.array(1)

    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)

    state = layer.update_state(state, parameters)
    parameters = layer.update_parameters(state, parameters)

    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
