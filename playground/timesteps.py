import time
import plastix as px
import random
import jax
import jax.numpy as jnp
import jax.lib.xla_bridge as xla

# pip install gputil
import GPUtil

# remove warning messages when running on CPU
if not GPUtil.getAvailable():
    jax.config.update("jax_platforms", "cpu")

# script running on CPU or the GPU?
print(xla.get_backend().platform)


def generate_random_edges(n, m, density):
    # note: this may also return the same edge multiple times,
    # but that's fine for this benchmarking test
    num_edges = int(density * n * m)
    edges = []
    for i in range(num_edges):
        edges.append((random.randint(0, n - 1), random.randint(0, m - 1)))

    return edges


n = 300
m = 300
time_steps = 10

edges = generate_random_edges(n, m, density=1)
sparse_layer = px.layers.SparseLayer(
    n, m, edges, px.kernels.edges.FixedWeight(), px.kernels.nodes.SumNonlinear()
)

sl_state = sparse_layer.init_state()
sl_parameters = sparse_layer.init_parameters()
sl_parameters.edges.weight *= 0.5
sl_state.input_nodes.rate = jnp.array([[0.0] for _ in range(n)])

start = time.time()

for t in range(time_steps):
    sparse_layer.update_state(sl_state, sl_parameters, use_jit=True)
    sparse_layer.update_parameters(sl_state, sl_parameters, use_jit=True)
