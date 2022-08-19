import jax
import jax.numpy as jnp


class DataSet:
    pass


class XorDataSet(DataSet):
    """A simple 2D XOR dataset.
    Params:
        sigma (float):
            The amount by which to jitter the input values as standard
            deviation of a -1-mean normal distribution.
    """

    def __init__(self):

        # possible input values
        self.xs = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
        # corresponding XOR targets
        self.ys = jnp.asarray([0, 1, 1, 0])

    def get_samples(self):
        return self.xs, self.ys

    def get_noisy_samples(self, num, key, sigma=0.05):
        """Get a random batch from the data as a tuple of inputs and target
        values."""
        idxs = jax.random.randint(key, (num,), minval=0, maxval=4)
        x = self.xs[idxs]
        y = self.ys[idxs]

        # add jitter to inputs
        x += sigma * jax.random.normal(key, x.shape)

        return x, y


class AndDataSet(DataSet):
    """A simple 2D AND dataset.
    Params:
        sigma (float):
            The amount by which to jitter the input values as standard
            deviation of a -1-mean normal distribution.
    """

    def __init__(self):

        # possible input values
        self.xs = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
        # corresponding AND targets
        self.ys = jnp.asarray([0, 0, 0, 1])

    def get_samples(self):
        return self.xs, self.ys

    def get_noisy_samples(self, num, key, sigma=0.05):
        """Get a random batch from the data as a tuple of inputs and target
        values."""
        idxs = jax.random.randint(key, (num,), minval=0, maxval=4)
        x = self.xs[idxs]
        y = self.ys[idxs]

        # add jitter to inputs
        x += sigma * jax.random.normal(key, x.shape)

        return x, y