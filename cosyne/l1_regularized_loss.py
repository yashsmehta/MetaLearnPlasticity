import os
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import jit
import optax
import time
import pandas as pd
import numpy as np
import time
import math
import random
from pathlib import Path
import cosyne.utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_gaussian(key, shape, scale=0.1):
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def generate_mask(plasticity_rule, num_meta_params):
    if plasticity_rule == "oja":
        assert num_meta_params >= 2, "number of meta-parameters must atleast be 2"
        mask = np.zeros((27,))
        idx_a0 = utils.powers_to_A_index(1, 1, 0)
        idx_a1 = utils.powers_to_A_index(0, 2, 1)
        idx = random.sample(
            [i for i in range(27) if i not in [idx_a0, idx_a1]], num_meta_params - 2
        )
        idx.extend([idx_a0, idx_a1])
        mask[idx] = 1

    else:
        raise Exception("currently masking is only implemented for Oja's rule")

    return jnp.array(mask)


@Partial(jax.jit, static_argnums=(0,))
def generate_A_teacher(plasticity_rule):
    A_teacher = np.zeros((27,))
    if plasticity_rule == "oja":
        A_teacher[utils.powers_to_A_index(1, 1, 0)] = 1
        A_teacher[utils.powers_to_A_index(0, 2, 1)] = -1

    elif plasticity_rule == "hebbian":
        A_teacher[utils.powers_to_A_index(1, 1, 0)] = 1

    elif plasticity_rule == "random":
        A_teacher[utils.powers_to_A_index(1, 1, 0)] = np.random.rand()
        A_teacher[utils.powers_to_A_index(0, 2, 1)] = -1 * np.random.rand()

    else:
        raise Exception("plasticity rule must be either oja, hebbian or random")

    return jnp.array(A_teacher)


def generate_weight_trajec(x, weights, A):
    weight_trajectory = []

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        weight_trajectory.append([w.copy() for w in weights])

    return weight_trajectory


def generate_activity_trajec(x, weights, A):
    activity_trajectory = []

    for i in range(len(x)):
        weights = update_weights(weights, x[i], A)
        act = forward(weights, x[i])
        activity_trajectory.append(act)

    return activity_trajectory


def calc_loss_weight_trajec(mask, l1_lmbda, weights, x, A, weight_trajectory):
    A = A * mask
    # add L1 regularization term to enforce sparseness
    # IMP: L1 regularizer needs to be calculated only on a subset of A + do correct loss normalization
    loss = l1_lmbda * jnp.sum(jnp.absolute(A))

    for i in range(len(weight_trajectory)):
        weights = update_weights(weights, x[i], A)
        teacher_weights = weight_trajectory[i]

        for j in range(len(weights)):
            loss += jnp.mean(optax.l2_loss(weights[j], teacher_weights[j]))

    return loss / len(weight_trajectory)


def calc_loss_activity_trajec(mask, l1_lmbda, weights, x, A, activity_trajectory):
    A = A * mask
    # add L1 regularization term to enforce sparseness
    loss = l1_lmbda * jnp.sum(jnp.absolute(A))
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


def update_weights_(mask, weights, x, A):
    act = forward(weights, x)
    A = A * mask
    for layer in range(len(weights)):
        dw = 0
        for index in range(len(A)):
            i, j, k = utils.A_index_to_powers(index)
            dw += A[index] * jnp.multiply(
                jnp.outer(act[layer + 1] ** j, act[layer] ** i), weights[layer] ** k
            )

        if dw.shape != weights[layer].shape:
            raise Exception(
                "dw and w should be of the same shape to prevent broadcasting while adding"
            )
        weights[layer] += dw

    return weights


def forward_(non_linear, weights, x):
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
        num_meta_params,
        l1_lmbda,
        sparsity,
        noise_scale,
        log_expdata,
        output_file,
        jobid,
    ) = utils.parse_args_old()

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

    key, key2 = jax.random.split(key)
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        # teacher_weights.append(generate_gaussian(key, (n, m), scale=1))
        # student_weights.append(generate_gaussian(key, (n, m), scale=1))
        teacher_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))
        student_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))

    A_teacher = generate_A_teacher(plasticity_rule)
    mask = generate_mask(plasticity_rule, num_meta_params)

    key, key2 = jax.random.split(key)
    A_student = generate_gaussian(key2, (27,), scale=1e-4)
    # A_student = jnp.zeros((27,))

    global forward, update_weights
    forward = jit(Partial(forward_, non_linear))
    update_weights = jit(Partial((update_weights_), mask))

    # same random initialization of the weights at the start for student and teacher network
    if type == "activity":
        calc_loss_trajec = jit(Partial((calc_loss_activity_trajec), mask, l1_lmbda))
        generate_trajec = jit(generate_activity_trajec)
    else:
        calc_loss_trajec = jit(Partial((calc_loss_weight_trajec), mask, l1_lmbda))
        generate_trajec = jit(generate_weight_trajec)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(A_student)

    expdata = {
        "A_" + str(i) + str(j) + str(k): []
        for i in range(3)
        for j in range(3)
        for k in range(3)
    }
    expdata.update(
        {"loss": [], "mean_grad_norm": [], "epoch": jnp.arange(meta_epochs + 1)}
    )

    start_time = time.time()

    for epoch in range(meta_epochs + 1):
        key = jax.random.PRNGKey(0)
        expdata["loss"].append(0)
        expdata["mean_grad_norm"].append(0)
        for idx in range(len(A_student)):
            pi, pj, pk = utils.A_index_to_powers(idx)
            expdata["A_{}{}{}".format(pi, pj, pk)].append(A_student[idx])

        for _ in range(num_trajec):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (len_trajec, input_dim), scale=0.1)
            trajectory = generate_trajec(x, teacher_weights, A_teacher)

            loss_T, grads = jax.value_and_grad(calc_loss_trajec, argnums=2)(
                student_weights, x, A_student, trajectory
            )
            # loss_T = calc_loss_trajec(student_weights, x, A_student, trajectory)

            expdata["mean_grad_norm"][-1] += jnp.linalg.norm(grads)
            expdata["loss"][-1] += loss_T

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

        expdata["loss"][-1] = round(math.sqrt(expdata["loss"][-1] / num_trajec), 6)
        expdata["mean_grad_norm"][-1] = round(
            math.sqrt(expdata["mean_grad_norm"][-1] / num_trajec), 6
        )

        print(
            "sqrt avg. avg. loss (across num_trajectories, len_trajectory)",
            expdata["loss"][-1],
        )
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
        df["device"],
        df["num_meta_params"],
        df["l1_lmbda"],
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
        num_meta_params,
        l1_lmbda,
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
