import numpy as np
from sklearn.decomposition import PCA
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax

import plastix as px

def mse_loss(layer, state, parameters, x, pcs_0):
    for i in range(len(x)):
        state.input_nodes.rate = x[i]
        state = layer.update_state(state, parameters)
        parameters = layer.update_parameters(state, parameters)

    return jnp.mean(optax.l2_loss(jnp.squeeze(parameters.edges.weight), pcs_0))

def generate_datasets_Gaussian(n_datasets, n_samples, n_input, D):
    #Xi taken from N(0,transp(Q)*D*Q)
    #D diagonal matrix, will be rotated to become the covariance matrix of the dataset
    
    X = np.zeros((n_datasets, n_samples, n_input))
    for d_num in range(n_datasets):
        A = np.random.rand(n_input, n_input)
        Q, _ = np.linalg.qr(A)  #generate a random rotation to apply to D 
        if np.linalg.det(Q) < 0:
            Q = -Q
        Cov = np.matmul(np.transpose(Q),np.matmul(D, Q)) 
        for s_num in range(n_samples): 
            X[d_num, s_num, :] = np.random.multivariate_normal([0 for i in range(n_input)], Cov)
    
    #compute & store principal vectors to compare to PCA network performance afterwards
    pcs = np.zeros((n_datasets, n_input, n_input)) #which dataset, which pc, which dim
    for d_num in range(n_datasets):
        pca = PCA()
        pca.fit(X[d_num, :, :])
        for i_num in range(n_input):
            pcs[d_num, i_num, :] = pca.components_[i_num]
    # x: [num_datasets, num_samples, num_input_neurons]
    # pcs: [num_datasets, num_input_neurons, num_input_neurons]
    return X, pcs

def main():
    key = jax.random.PRNGKey(0)
    meta_epochs = 5 

    D3 = np.diag([1,0.5,0])
    D5 = np.diag([1,0.7,0.2,0.1,0])
    D100 = np.zeros((100,100)); D100[0,0] = 1; D100[1,1] = 0.9; D100[2,2] = 0.8; D100[3,3] = 0.7; D100[4,4] = 0.6; D100[5,5] = 0.5
    
    x, pcs = generate_datasets_Gaussian(n_datasets=meta_epochs, n_samples=100, n_input=5, D=D5)

    x = jnp.array(x) 
    layer = px.layers.DenseLayer(
        5,
        1,
        px.kernels.edges.RatePolynomialUpdateEdge(),
        px.kernels.nodes.SumLinear(),
    )
    state = layer.init_state()
    parameters = layer.init_parameters()
    print("A:", parameters.edges.coefficient_matrix) 
    parameters.edges.coefficient_matrix = 0.01 * jax.random.normal(key, (3,3,3))
    print("A:", parameters.edges.coefficient_matrix) 

    state = layer.update_state(state, parameters)
    # change loss to mse on PCA of the input neurons
    loss = Partial((mse_loss), layer)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(parameters.edges.coefficient_matrix)

    for e in range(meta_epochs):
        key, _ = jax.random.split(key)
        grads = jax.grad(loss, argnums=1)(state, parameters, x[e], pcs[e][0])
        updates, opt_state = optimizer.update(grads.edges.coefficient_matrix, opt_state, parameters.edges.coefficient_matrix)
        parameters.edges.coefficient_matrix = optax.apply_updates(parameters.edges.coefficient_matrix, updates)

    state = layer.update_state(state, parameters)
    print("edge parameters:", parameters.edges.weight)
    print("edge state:", state.edges.signal)
    print("prediction: ", state.output_nodes.rate)


if __name__ == "__main__":
    main()
