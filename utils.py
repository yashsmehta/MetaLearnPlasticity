import jax


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)

def generate_random_connectivity(key, neurons, sparsity):
    """
    returns a square connectivity matrix of shape [neurons x neurons]
    with a specific sparsity of connections
    """

    return jax.random.bernoulli(key, p=sparsity, shape=(neurons, neurons))