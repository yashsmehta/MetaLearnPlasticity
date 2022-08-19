import time
import jax
import jax.numpy as jnp
import GPUtil
import jax.lib.xla_bridge as xla
import numpy as np


def aggregate(values):
    return jnp.sum(values)


def aggregate_all(m, values, edge_index_range):
    return jnp.array(
        [
            aggregate(values.at[edge_index_range[j][0] : edge_index_range[j][1]].get())
            # aggregate(values[edge_index_range[j][0]:edge_index_range[j][1]])
            for j in range(m)
        ]
    )


def old_aggregate_all(values, indices):
    return jnp.array([aggregate(values[jnp.array(inds)]) for inds in indices])


if not GPUtil.getAvailable():
    jax.config.update("jax_platforms", "cpu")

# script running on CPU or the GPU?
print(xla.get_backend().platform)

if __name__ == "__main__":
    # number of output nodes
    m = n = 500
    density = 0.1
    num_edges = int(density * m * n)
    values = jnp.array(np.random.rand(num_edges), jnp.float32)
    print("output nodes: ", m)
    print("number of edges: ", num_edges)

    edge_indices = []
    for j in range(m):
        in_edges_j = np.random.randint(1, m - 1)
        edge_indices.append(tuple(np.random.randint(0, m - 1, in_edges_j)))

    edge_indices = tuple(edge_indices)
    # print(edge_indices)

    jit_aggregate_all = jax.jit(old_aggregate_all, static_argnames=("indices",))
    start = time.time()
    c = jit_aggregate_all(values, edge_indices)
    print("compilation time (random edge indices)", time.time() - start)

    jit_aggregate_all = jax.jit(
        aggregate_all, static_argnames=("edge_index_range", "m")
    )
    # range: [,)
    edge_index_range = []

    for j in range(m):
        edge_index_range.append((int(n * density * j), int(n * density * (j + 1))))

    edge_index_range = tuple(edge_index_range)

    start = time.time()
    c = jit_aggregate_all(m, values, edge_index_range)
    print("compilation time (continuous indices)", time.time() - start)

    # print(edge_indices)

    # start = time.time()
    # c = jit_aggregate_all(values, edge_index_range)
    # print("compilation time (random edge indices)", time.time() - start)

    # start = time.time()
    # for i in range(1000):
    #     c = jit_aggregate_all(values, edge_indices)
    # print("jit run time", time.time() - start)
