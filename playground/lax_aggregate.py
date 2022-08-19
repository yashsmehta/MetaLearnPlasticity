import time
import jax
import jax.numpy as jnp
import numpy as np
import random

jax.config.update("jax_platforms", "cpu")


def aggregate(es):
    return jnp.sum(es)


def for_aggregate_all(edge_states, indices):
    return jnp.array(
        [
            aggregate(
                edge_states[
                    jnp.array(indices[j])
                    if (indices[j])
                    else jnp.array([], dtype=jnp.int32)
                ]
            )
            for j in range(len(indices))
        ]
    )


# use output_node_parameters
# use: .at() & .set()
def lax_body_func(j, val):
    edge_states, indices = val
    j += 1
    return aggregate(edge_states[jnp.array(indices[j], jnp.int32)])


def cond_func(j, val):
    return j < 10


if __name__ == "__main__":

    edge_states = jnp.array([1, 2, 3, 100, 3, 4, 5, 6], jnp.float32)
    # number of output nodes
    n, m = 500, 500
    density = 1
    total_edges = remaining_edges = int(n * m * density)

    print("total_edges:", total_edges)
    print("output nodes:", m)

    # generate random edge indices
    indices = []
    for _ in range(m):
        randi = random.randint(0, min(remaining_edges, m))
        remaining_edges -= randi
        e = tuple(random.sample(range(total_edges), randi))
        indices.append(e)

    # print("list indices:", indices)
    indices = tuple(indices)
    # print("tuple indices:", indices)

    jit_aggregate_all = jax.jit(for_aggregate_all, static_argnames=("indices",))

    start = time.time()
    c = jit_aggregate_all(edge_states, indices)
    print("compile time    : %f.3s" % (time.time() - start))

    # for i in range(5):
    # c = jit_aggregate_all(edge_states, indices)

    # print("jit    : %f.3s" % (time.time() - start))

    # start = time.time()
    # for i in range(1):
    #     c = jnp.array([jax.lax.fori_loop(0, 5, lax_body_func, (edge_states, indices))])

    # cond_fun = lambda j, val: j<10

    # j=0
    # val = edge_states, indices
    # c = [jax.lax.while_loop(cond_fun, lax_body_func, (j, val))]
    # print("lax    : %f.3s" % (time.time() - start))
