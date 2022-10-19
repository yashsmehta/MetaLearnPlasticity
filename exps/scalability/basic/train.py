import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
import time
import pandas as pd
import time
import math
from pathlib import Path

import utils


def generate_gaussian(key, shape, scale=0.1):
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


@jax.jit
def generate_weight_trajec(x, weights, A):
    weight_trajectory = []

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        weight_trajectory.append([w.copy() for w in weights])
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

    for i in range(len(weight_trajectory)):
        weights = update_weights(weights, x[i], A)
        teacher_weights = weight_trajectory[i]

        for j in range(len(weights)):
            loss += jnp.mean(optax.l2_loss(weights[j], teacher_weights[j]))

    return loss / len(weight_trajectory)


@jax.jit
def calc_loss_activity_trajec(weights, x, A, activity_trajectory):
    loss = 0
    use_input = True

    for i in range(len(activity_trajectory)):
        loss_t = []
        weights = update_weights(weights, x[i], A)
        act = forward(weights, x[i])
        teacher_act = activity_trajectory[i]
        for j in range(len(act)):
            loss_t.append(jnp.mean(optax.l2_loss(act[j], teacher_act[j])))
        if not use_input:
            loss_t.pop(0)
        loss += sum(loss_t)

    return loss / len(activity_trajectory)


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
        output_file,
        jobid,
    ) = utils.parse_args()

    key = jax.random.PRNGKey(jobid)

    device = jax.lib.xla_bridge.get_backend().platform  # are we running on CPU or GPU?

    path = "explogs/"
    layer_sizes = [input_dim]

    for _ in range(hidden_layers):
        layer_sizes.append(hidden_neurons)
        if hidden_neurons == -1:
            raise Exception("specify the number of hidden neurons in the network")

    layer_sizes.append(output_dim)
    print("network architecture: ", layer_sizes)
    print("platform: ", device)
    teacher_weights, student_weights = [], []

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        teacher_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))
        student_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))

    if plasticity_rule == "oja":
        A_teacher = jnp.array([1.0, -1.0])
    elif plasticity_rule == "hebbian":
        A_teacher = jnp.array([1.0, 0])
    elif plasticity_rule == "random":
        A_teacher = generate_gaussian(key, (2,), scale=1)
    else:
        raise Exception("plasticity rule must be either oja, hebbian or random")

    key, key2 = jax.random.split(key)
    # A_student = generate_gaussian(key2, (2,), scale=1e-3)
    # A_student = jnp.array([1., -1.])
    A_student = jnp.zeros((2,))

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

    expdata = {"A_0": [], "A_1": [], "loss": [], "grads_norm": []}
    expdata["epoch"] = jnp.arange(meta_epochs)
    start_time = time.time()
    print("[init] A teacher", A_teacher)
    print("[init] A student", A_student)

    for epoch in range(meta_epochs):
        key = jax.random.PRNGKey(0)
        expdata["loss"].append(0.0)
        expdata["grads_norm"].append([])

        for _ in range(num_trajec):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (len_trajec, input_dim), scale=0.1)
            trajectory = generate_trajec(x, teacher_weights, A_teacher)

            loss_T, grads = jax.value_and_grad(calc_loss_trajec, argnums=2)(
                student_weights, x, A_student, trajectory
            )
            # loss_T = calc_loss_trajec(student_weights, x, A_student, trajectory)

            print("grads: ", grads)
            exit()
            expdata["grads_norm"][-1].append(jnp.linalg.norm(grads))
            expdata["loss"][-1] += loss_T

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

        expdata["loss"][-1] = round(math.sqrt(expdata["loss"][-1] / num_trajec), 6)
        expdata["grads_norm"][-1] = [
            round(math.sqrt(grad_norm_t), 6)
            for grad_norm_t in expdata["grads_norm"][-1]
        ]
        expdata["A_0"].append(A_student[0])
        expdata["A_1"].append(A_student[1])

        print("A student:", A_student)
        print(
            "sqrt avg. avg. loss (across num_trajectories, len_trajectory)",
            expdata["loss"][-1],
        )
        print()

    print("note: logs store sqrt of loss & gradient norm")
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
        df["device"],
        df["num_meta_params"],
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
        device,
        len(A_student),
    )

    print(df.head(5))

    if log_expdata:
        use_header = False
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "{}.csv".format(output_file)):
            use_header = True

        df.to_csv(path + "{}.csv".format(output_file), mode="a", header=use_header)
        print("wrote training logs to disk")


if __name__ == "__main__":
    main()
