from importlib.metadata import metadata
import numpy as np
import plastix as px
import jax
import jax.numpy as jnp
import dataloaders.associative_learning_ds as alds
import dataloaders.toy_ds as tds
import dataloaders.familiarity_detection_ds as fdds


def main():
    # meta_data = dict()
    # meta_data['prob_valid'] = 0.5
    # meta_data['B'], meta_data['f'] = 1, 0.2
    # s,u,rtarg,trialinfo = ds.genrandtrials(meta_data)
    # print(s,u,rtarg,trialinfo)
    # layer = px.layers.SparseLayer(2, 1, edges, px.kernels.edges.STDPUpdateEdge(), px.kernels.nodes.DifferentialNonlinear())
    # state = layer.init_state()
    # parameters = layer.init_parameters()

    # parameters.edges.dt*=0.5
    # parameters.edges.wmax*=0.05
    # parameters.edges.tau*=5
    # parameters.edges.tauw*=5
    # parameters.edges.stdpfac*=1

    # print("edge parameters:", parameters.edges.weight)
    # print("edge state:", state.edges.signal)
    # print("delta t:", parameters.edges.dt)
    # print("dopamine", parameters.edges.dopamine)
    # print("prediction: ", state.output_nodes.rate)

    edges = [(0, 0), (1, 0), (2, 0)]
    # s,u,rtarg,trialinfo = alds.genrandtrials()
    data = fdds.generate_recog_data(T=5, d=10)
    print(data)

    # print(s, u, rtarg)
    # print("trial info", trialinfo)
    layer = px.layers.SparseLayer(
        3, 1, edges, px.kernels.edges.FixedWeightEdge(), px.kernels.nodes.SumNonlinear()
    )


if __name__ == "__main__":
    main()
