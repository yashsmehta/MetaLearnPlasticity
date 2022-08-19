import numpy as np
import plastix as px
import jax
import jax.numpy as jnp
import pandas as pd


def main():
    edges = [(0, 0), (1, 0), (2, 0)]

    layer = px.layers.SparseLayer(
        3,
        1,
        edges,
        px.kernels.edges.RatePolynomialUpdateEdge(),
        px.kernels.nodes.SumNonlinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()

    key = jax.random.PRNGKey(seed=0)
    state.input_nodes.rate = jax.random.uniform(key, (3, 1), minval=0, maxval=1)
    parameters.edges.weight = jax.random.uniform(key, (3, 1), minval=0, maxval=0.1)
    parameters.edges.lr = jnp.array(0.01)

    print("init edge parameters:", parameters.edges.weight)
    print("init edge state:", state.edges.signal)
    epochs = 20
    weights = {"w0": [], "w1": [], "w2": []}

    for e in range(epochs):
        state = layer.update_state(state, parameters)
        parameters = layer.update_parameters(state, parameters)
        print("params:", parameters.edges.weight)
        weights["w0"].append(jnp.squeeze(parameters.edges.weight[0]))
        weights["w1"].append(jnp.squeeze(parameters.edges.weight[1]))
        weights["w2"].append(jnp.squeeze(parameters.edges.weight[2]))

    print("Training done!")
    df = pd.DataFrame.from_dict(weights)
    df.to_csv(
        "/groups/funke/home/mehtay/research/connectome_modelling/run/expdata/trajectory.csv"
    )


if __name__ == "__main__":
    main()
