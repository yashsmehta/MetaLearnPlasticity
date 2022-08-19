import numpy as np
import networkx as nx
import pandas as pd
import plastix as px
import time

import neuprint
from neuprint import Client
from neuprint import NeuronCriteria as NC
from neuprint import fetch_neurons
from neuprint import fetch_adjacencies
from neuprint import merge_neuron_properties
from neuprint import fetch_roi_hierarchy


def main():
    start_time = time.time()
    # register neuprint account, and get personal token
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Inlhc2hzbWVodGE5NUBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdqdnJKbW9vR1VqZHFDTzZCT3pVSGRMM2dZX1RDd2h4QnN6RDlDclhoST1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgzNzkwMTU0M30.aJ_od2MDWIbqAh8c8Orx9TjbvCvsiH-v23TxC3dMg6M"
    c = Client("neuprint.janelia.org", "hemibrain:v1.2.1", TOKEN)

    # define lists of neuron instances
    KCs = ["KCab-p_R", "KCab-c_R", "KCab-s_R"]
    DANs = ["PAM11(a1)_R"]
    MBONs = ["MBON07(a1)_R"]

    neuron_df, conn_df = fetch_adjacencies(
        # NC(instance=KCs + DANs + MBONs), NC(instance=KCs + DANs + MBONs)
        NC(instance=DANs + MBONs),
        NC(instance=DANs + MBONs),
    )

    # treat synaptic connection between neurons irrespective of the region of connection (new weight is the sum of weights of individual regions)
    conn_df = (
        conn_df.groupby(["bodyId_pre", "bodyId_post"])["weight"]
        .apply(np.sum)
        .reset_index()
    )
    conn_df = get_functional_connections(conn_df, synapse_threshold=1)
    conn_df.weight /= conn_df["weight"].max(axis=0)
    print("time to load neuprint data: {}s".format(round(time.time() - start_time, 2)))
    G = nx.from_pandas_edgelist(
        conn_df,
        source="bodyId_pre",
        target="bodyId_post",
        edge_attr=["weight"],
        create_using=nx.DiGraph(),
    )
    print("creating networkx graph: ")
    print("#nodes: ", G.number_of_nodes())
    print("#edges: ", G.number_of_edges())
    print()

    G = nx.convert_node_labels_to_integers(G)
    edges = list(G.edges())
    edges = sorted(edges, key=lambda x: x[1])
    m = n = G.number_of_nodes()
    layer = px.layers.SparseLayer(
        n,
        m,
        edges,
        px.kernels.edges.FixedWeightEdge(),
        px.kernels.nodes.SumNonlinear(),
    )
    print("created plastix layer")
    layer_state = layer.init_state()
    layer_parameters = layer.init_parameters()
    print("initial layer weights", layer_parameters.edges.weight[:5])
    init_layer_with_netx(layer_parameters, edges, G)
    print("connectome initialized layer weights", layer_parameters.edges.weight[:5])


def init_layer_with_netx(parameters, edges, G):
    # shape of w_init (num_edges, weight parameter shape)
    w_init = np.zeros(parameters.edges.weight.shape)
    for i, (u, v) in enumerate(edges):
        w_init[i] = G.get_edge_data(u, v)["weight"]

    parameters.edges.weight = w_init
    return


def get_functional_connections(conn_df, synapse_threshold=1):
    # remove connections lesser than 'x' number of synapses
    drop = []

    for ind in conn_df.index:
        if conn_df["weight"][ind] < synapse_threshold:
            drop.append(ind)

    print("all edges:", len(conn_df))
    print("remaining edges:", len(conn_df) - len(drop))

    return conn_df.drop(drop, axis=0)


if __name__ == "__main__":
    main()
