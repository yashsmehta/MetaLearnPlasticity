import jax


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def generate_random_connectivity(key, m, n, sparsity):
    """
    returns a random binary connectivity mask of shape [M x N]
    with a specific connectivity sparseness
    """

    return jax.random.bernoulli(key, p=float(sparsity), shape=(m, n))

def create_nested_list(num_outer, num_inner):
    return [[[] for _ in range(num_inner)] for _ in range(num_outer)]