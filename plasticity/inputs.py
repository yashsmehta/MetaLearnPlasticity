import jax


def sample_inputs(mus, sigmas, k, random_key):

    # get a normally distributed variable
    x = jax.random.normal(random_key, shape=mus[0].shape)

    # shift and scale according to mus[k]
    x = x + mus[k]
    x = x @ sigmas[k]

    return x
