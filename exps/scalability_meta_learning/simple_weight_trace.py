import jax
import jax.numpy as jnp
import optax


def generate_gaussian(key, shape, scale=0.1):
    return scale * jax.random.normal(key, (shape))


def generate_weight_trajectory(x, weights, A):
    weight_trajectory = []

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        weight_trajectory.append(weights)

    return weight_trajectory


def trajectory_loss(x, weights, A, weight_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        loss += jnp.mean(optax.l2_loss(weights, weight_trajectory[i]))
    return loss


def update_weights(x, weights, A):
    y = jnp.dot(x, weights)
    dw = A[0] * x * y + A[1] * jnp.multiply(y**2, weights)
    weights += dw
    return weights


def main():
    key = jax.random.PRNGKey(0)
    meta_epochs = 1
    num_trajectories = 5
    length = 5
    m, n = 5, 1
    teacher_weights = generate_gaussian(key, (m, n), scale=1 / (m + n))
    student_weights = generate_gaussian(key, (m, n), scale=1 / (m + n))
    A_teacher = jnp.array([1, -1])
    A_student = jnp.zeros((2,))
    key, _ = jax.random.split(key)

    # same random initialization of the weights at the edges for student and teacher network

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(A_student)

    for epoch in range(meta_epochs):
        key = jax.random.PRNGKey(0)
        for _ in range(num_trajectories):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (length, m), scale=0.1)
            weight_trajectory = generate_weight_trajectory(
                x, teacher_weights, A_teacher
            )
            # print("weight trajectory", weight_trajectory)
            loss_t, grads = jax.value_and_grad(trajectory_loss, argnums=2)(
                x, student_weights, A_student, weight_trajectory
            )
            print("loss: ", loss_t)
            print("grads A: ", grads)
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

    print("A final:", A_student)


if __name__ == "__main__":
    main()
