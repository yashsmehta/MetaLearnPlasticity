import jax
import jax.numpy as jnp
import optax
import time
import pandas as pd
from pathlib import Path
import os

import utils

def generate_gaussian(key, shape, scale=0.1):
    return scale * jax.random.normal(key, (shape))

@jax.jit
def generate_weight_trajec(x, weights, A):
    weight_trajectory = []

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        weight_trajectory.append(weights)

    return weight_trajectory

@jax.jit
def generate_activity_trajec(x, weights, A):
    activity_trajectory = []

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        y = forward_pass(x[i], weights)
        activity_trajectory.append(y)

    return activity_trajectory

@jax.jit
def calc_loss_weight_trajec(x, weights, A, weight_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        loss += jnp.mean(optax.l2_loss(weights, weight_trajectory[i]))
    return loss

@jax.jit
def calc_loss_activity_trajec(x, weights, A, activity_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        y = forward_pass(x[i], weights)
        loss += jnp.mean(optax.l2_loss(y, activity_trajectory[i]))
    return loss

def update_weights(x, weights, A):
    y = forward_pass(x, weights)
    dw = A[0] * x * y + A[1] * jnp.multiply(y**2, weights)
    weights += dw
    return weights

def forward_pass(x, weights):
    return jax.nn.sigmoid(jnp.dot(x, weights))

"""
document the following:
    m/n neurons
    trajectory update time
    # learnable params
    underlying update rule
    type: activity / weight trace
    non-linearity
"""

def main():
    (
        m, 
        n, 
        non_linearity,
        plasticity_rule,
        meta_epochs,
        num_trajec,
        len_trajec,
        type,
        log_expdata,
        jobid,
    ) = utils.parse_args()

    key = jax.random.PRNGKey(jobid)
    path = "explogs/"
    teacher_weights = generate_gaussian(key, (m, n), scale=1 / (m + n))
    student_weights = generate_gaussian(key, (m, n), scale=1 / (m + n))
    A_teacher = jnp.array([1, -1])
    A_student = jnp.zeros((2,))
    key, _ = jax.random.split(key)
    config = {}
    config['type'] = 'activity'

    # same random initialization of the weights at the start for student and teacher network
    if(config['type'] == 'activity'):
        calc_loss_trajec = calc_loss_activity_trajec
        generate_trajec = generate_activity_trajec
    else:
        calc_loss_trajec = calc_loss_weight_trajec
        generate_trajec = generate_weight_trajec


    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(A_student)

    expdata = {'A_0':[], 'A_1':[], 'loss':[]} 
    expdata['epoch'] = jnp.arange(meta_epochs)

    for epoch in range(meta_epochs):
        start_time = time.time()
        key = jax.random.PRNGKey(0)
        expdata['loss'].append(0.)

        for _ in range(num_trajec):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (len_trajec, m), scale=0.1)
            trajectory = generate_trajec(
                x, teacher_weights, A_teacher
            )
            # print("weight trajectory", weight_trajectory)
            loss_t, grads = jax.value_and_grad(calc_loss_trajec, argnums=2)(
                x, student_weights, A_student, trajectory
            )
            expdata['loss'][-1]+=loss_t

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

        expdata['A_0'].append(A_student[0])
        expdata['A_1'].append(A_student[1])

        print("A student:", A_student)
        print("loss (all trajectories):", expdata['loss'][-1])
        print("epoch {} time: {}s".format(epoch, time.time() - start_time))
        print()

    df = pd.DataFrame(expdata)
    if log_expdata:
        use_header = False
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "ml-plasticity.csv"):
            use_header = True

        df.to_csv(path + "ml-plasticity.csv", mode="a", header=use_header)
        print("wrote training logs to disk")


if __name__ == "__main__":
    main()
