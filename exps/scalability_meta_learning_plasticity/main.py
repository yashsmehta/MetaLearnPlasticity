import jax
import jax.numpy as jnp
import optax
import time
import pandas as pd
from pathlib import Path
import os
from jax.tree_util import Partial

import utils


def generate_gaussian(key, shape, scale=0.1):
    return scale * jax.random.normal(key, (shape))


@jax.jit
def generate_weight_trajec(x, weights, A):
    weight_trajectory = []

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        weight_trajectory.append(weights)

    return weight_trajectory


@jax.jit
def generate_activity_trajec(x, weights, A):
    activity_trajectory = []

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        act = forward(weights, x[i])
        activity_trajectory.append(act)

    return activity_trajectory


@jax.jit
def calc_loss_weight_trajec(weights, x, A, weight_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        teacher_weights = weight_trajectory[i]
        for j in range(len(weights)):
            loss += jnp.mean(optax.l2_loss(weights[j], teacher_weights[j]))
    return loss


@jax.jit
def calc_loss_activity_trajec(weights, x, A, activity_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        act = forward(weights, x[i])
        teacher_act = activity_trajectory[i]
        for j in range(len(act)):
            loss += jnp.mean(optax.l2_loss(act[j], teacher_act[j]))
    return loss


def update_weights(weights, x, A):
    act = forward(weights, x)
    for layer in range(len(weights)):
        dw = (
            A[0] * jnp.outer(act[layer + 1], act[layer])
            + A[1] * weights[layer] * act[layer + 1] ** 2
        )
        if dw.shape != weights[layer].shape:
            raise Exception(
                "dw and w should be of the same shape to prevent broadcasting while adding"
            )
        weights[layer] += dw

    return weights


def network_forward(non_linear, weights, x):
    act = [jnp.expand_dims(x, 1)]
    for layer in range(len(weights)):
        h = jnp.dot(weights[layer], act[-1])
        if non_linear:
            act.append(jax.nn.sigmoid(h))
        else:
            act.append(h)
    return act


def main():
    (
        input_dim,
        output_dim,
        hidden_layers,
        hidden_neurons,
        non_linear,  # True/False
        plasticity_rule,  # Hebb, Oja, Random
        meta_epochs,
        num_trajec,
        len_trajec,
        type,  # activity trace, weight trace
        log_expdata,
        jobid,
    ) = utils.parse_args()

    # key = jax.random.PRNGKey(jobid)
    key = jax.random.PRNGKey(0)
    path = "explogs/"
    layer_sizes = [input_dim]
    for _ in range(hidden_layers):
        layer_sizes.append(hidden_neurons)
    layer_sizes.append(output_dim)
    print("network architecture: ", layer_sizes)
    teacher_weights, student_weights = [], []

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        teacher_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))
        student_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))

    if plasticity_rule == "oja":
        A_teacher = jnp.array([1, -1])
    elif plasticity_rule == "hebbian":
        A_teacher = jnp.array([1, 0])
    elif plasticity_rule == "anti_hebbian":
        A_teacher = jnp.array([-1, 0])
    else:
        raise Exception("plasticity rule must be either oja, hebbian, anti_hebbian")

    A_student = jnp.zeros((2,))
    key, _ = jax.random.split(key)
    global forward
    forward = Partial((network_forward), non_linear)
    # same random initialization of the weights at the start for student and teacher network
    if type == "activity":
        calc_loss_trajec = calc_loss_activity_trajec
        generate_trajec = generate_activity_trajec
    else:
        calc_loss_trajec = calc_loss_weight_trajec
        generate_trajec = generate_weight_trajec

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(A_student)

    expdata = {"A_0": [], "A_1": [], "loss": []}
    expdata["epoch"] = jnp.arange(meta_epochs)
    start_time = time.time()

    print("A teacher", A_teacher)
    for epoch in range(meta_epochs):
        key = jax.random.PRNGKey(0)
        expdata["loss"].append(0.0)

        for _ in range(num_trajec):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (len_trajec, input_dim), scale=0.1)
            trajectory = generate_trajec(x, teacher_weights, A_teacher)
            # print("weight trajectory", trajectory)
            loss_t, grads = jax.value_and_grad(calc_loss_trajec, argnums=2)(
                student_weights, x, A_student, trajectory
            )
            expdata["loss"][-1] += loss_t

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

        expdata["A_0"].append(A_student[0])
        expdata["A_1"].append(A_student[1])

        print("A student:", A_student)
        print("cumilative loss", expdata["loss"][-1])
        print()

    avg_backprop_time = round(
        (time.time() - start_time) / (meta_epochs * num_trajec), 3
    )
    df = pd.DataFrame(expdata)
    pd.set_option("display.max_columns", None)

    (
        df["input_dim"],
        df["output_dim"],
        df["hidden_layers"],
        df["hidden_neurons"],
        df["non_linear"],
        df["plasticity_rule"],
        df["meta_epochs"],
        df["num_trajec"],
        df["len_trajec"],
        df["type"],
        df["jobid"],
        df["avg_backprop_time"],
    ) = (
        input_dim,
        output_dim,
        hidden_layers,
        hidden_neurons,
        non_linear,
        plasticity_rule,
        meta_epochs,
        num_trajec,
        len_trajec,
        type,
        jobid,
        avg_backprop_time,
    )

    print(df.head(5))

    if log_expdata:
        use_header = False
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "ml-plasticity.csv"):
            use_header = True

        df.to_csv(path + "ml-plasticity.csv", mode="a", header=use_header)
        print("wrote training logs to disk")


if __name__ == "__main__":
    main()
