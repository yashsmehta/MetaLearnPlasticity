import numpy as np
from sklearn.decomposition import PCA
import jax
import jax.numpy as jnp
import optax

import plastix as px


def generate_datasets_Gaussian(n_datasets, n_samples, n_input, D):
    # xi taken from N(0,transp(Q)*D*Q)
    # D diagonal matrix, will be rotated to become the covariance matrix of the dataset

    x = np.zeros((n_datasets, n_samples, n_input))
    for d_num in range(n_datasets):
        A = np.random.rand(n_input, n_input)
        Q, _ = np.linalg.qr(A)  # generate a random rotation to apply to D
        if np.linalg.det(Q) < 0:
            Q = -Q
        Cov = np.matmul(np.transpose(Q), np.matmul(D, Q))
        for s_num in range(n_samples):
            x[d_num, s_num, :] = np.random.multivariate_normal(
                [0 for i in range(n_input)], Cov
            )

    # compute & store principal vectors to compare to PCA network performance afterwards
    pcs = np.zeros(
        (n_datasets, n_input, n_input)
    )  # which dataset, which pc, which dim
    for d_num in range(n_datasets):
        pca = PCA()
        pca.fit(x[d_num, :, :])
        for i_num in range(n_input):
            pcs[d_num, i_num, :] = pca.components_[i_num]
    # shape of x: [num_datasets, num_samples, num_input_neurons]
    # shape of pcs: [num_datasets, (i-th principal component) num_input_neurons, num_input_neurons]
    return jnp.array(x), jnp.array(pcs)


def main():
    key = jax.random.PRNGKey(0)
    n_datasets = 1

    D3 = np.diag([1, 0.5, 0])
    D5 = np.diag([1, 0.7, 0.2, 0.1, 0])
    n_samples = 100
    x, pcs = generate_datasets_Gaussian(
        n_datasets=n_datasets, n_samples=n_samples, n_input=5, D=D5
    )

    layer = px.layers.DenseLayer(
        5,
        1,
        px.kernels.edges.RatePolynomialUpdateEdge(),
        px.kernels.nodes.SumLinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()
    A = np.zeros((3, 3, 3))
    A[1][1][0] = 1
    A[0][2][1] = -1
    parameters.edges.coefficient_matrix = jnp.array(A)

    state = layer.update_state(state, parameters)
    max_iter = 10
    print("learning rate:", parameters.edges.lr)

    for i in range(n_datasets):
        state = layer.init_state()
        parameters = layer.init_parameters()
        parameters.edges.coefficient_matrix = jnp.array(A)
        print("dataset: ", i)
        for e in range(max_iter):
            e = e % n_samples
            state.input_nodes.rate = x[i][e]
            state = layer.update_state(state, parameters)
            parameters = layer.update_parameters(state, parameters)
            loss = jnp.mean(
                optax.l2_loss(jnp.squeeze(parameters.edges.weight), pcs[i][0])
            )
            print("mse of weights from PCA1", loss)
        print()
        print()

    state = layer.update_state(state, parameters)
    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
