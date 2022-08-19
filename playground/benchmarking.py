import os
from pathlib import Path
import time
import plastix as px
import random
import jax.numpy as jnp
import jax.lib.xla_bridge as xla

# pip install gputil
# import GPUtil
import pandas as pd
import argparse

# remove warning messages when running on CPU
# if not GPUtil.getAvailable():
#     jax.config.update("jax_platforms", "cpu")

# script running on CPU or the GPU?
print(xla.get_backend().platform)

expdata = {}
parser = argparse.ArgumentParser()

parser.add_argument("--output_nodes", type=int, default=10)
parser.add_argument("--time_steps", type=int, default=10)
parser.add_argument("--density", type=float, default=0.1)

args = parser.parse_args()
print(args)
expdata["output_nodes"] = m = args.output_nodes
expdata["time_steps"] = time_steps = args.time_steps
expdata["density"] = density = args.density


def generate_random_edges(n, m, density):
    # note: this may also return the same edge multiple times,
    # but that's fine for this benchmarking test
    num_edges = int(density * n * m)
    edges = []
    for i in range(num_edges):
        edges.append((random.randint(0, n - 1), random.randint(0, m - 1)))

    return edges


dense_layer = px.layers.DenseLayer(
    m, m, px.kernels.edges.FixedWeight(), px.kernels.nodes.SumNonlinear()
)

edges = generate_random_edges(m, m, density=0.1)
sparse_layer = px.layers.SparseLayer(
    m, m, edges, px.kernels.edges.FixedWeight(), px.kernels.nodes.SumNonlinear()
)

# benchmark sparse layer implementation
start = time.time()
sl_state = sparse_layer.init_state()
sl_parameters = sparse_layer.init_parameters()
sl_parameters.edges.weight *= 0.5
sl_state.input_nodes.rate = jnp.array([[0.0] for _ in range(m)])
print("--- SPARSE LAYER --- ")
expdata["init_time"] = [time.time() - start]
print("initialization time", time.time() - start)

# start = time.time()
# sparse_layer.update_state(sl_state, sl_parameters, use_jit=False)
# sparse_layer.update_parameters(sl_state, sl_parameters, use_jit=False)
# print("non-jit compilation time", time.time() - start)
start = time.time()
sparse_layer.update_state(sl_state, sl_parameters, use_jit=True)
sparse_layer.update_parameters(sl_state, sl_parameters, use_jit=True)

expdata["compile_time"] = [time.time() - start]
print("jit compilation time", time.time() - start)
start = time.time()

for i in range(time_steps):
    sparse_layer.update_state(sl_state, sl_parameters, use_jit=True)
    sparse_layer.update_parameters(sl_state, sl_parameters, use_jit=True)

expdata["run_time"] = [time.time() - start]
print("run time", time.time() - start)


# benchmark dense layer implementation
start = time.time()
dl_state = dense_layer.init_state()
dl_parameters = dense_layer.init_parameters()
dl_parameters.edges.weight *= 0.5
dl_state.input_nodes.rate = jnp.array([[0.0] for _ in range(m)])
print("--- DENSE LAYER --- ")
expdata["init_time"].append(time.time() - start)
print("initialization time", time.time() - start)

start = time.time()
dense_layer.update_state(dl_state, dl_parameters, use_jit=True)
dense_layer.update_parameters(dl_state, dl_parameters, use_jit=True)
expdata["compile_time"].append(time.time() - start)
print("jit compilation time", time.time() - start)
start = time.time()

for i in range(time_steps):
    dense_layer.update_state(dl_state, dl_parameters, use_jit=True)
    dense_layer.update_parameters(dl_state, dl_parameters, use_jit=True)

expdata["run_time"].append(time.time() - start)
print("run time", time.time() - start, 2 * "\n")


expdata["output_nodes"] = [m] * 2
expdata["time_steps"] = [time_steps] * 2
expdata["density"] = [density, 1.0]
expdata["layer_type"] = ["sparse", "dense"]
expdata["platform"] = [xla.get_backend().platform] * 2

# df = pd.DataFrame.from_dict(expdata)
# use_header = False
# path = 'playground/expdata/'
# Path(path).mkdir(parents=True, exist_ok=True)
# if not os.path.exists(path + "benchmark-data.csv"):
#     use_header = True

# df.to_csv(path + "benchmark-data.csv", mode="a", header=use_header)
