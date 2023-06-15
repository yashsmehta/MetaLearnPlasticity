import numpy as np
from pprint import pprint

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# expected reward based update rule
def update_weights(x, reward, reward_history, weights):
    exp_reward = calculate_exp_reward(reward_history, window=10)
    dw = x * (reward - exp_reward)
    print("expected reward: ", exp_reward)
    print(f"weights: {weights}")
    print(f"dW: {dw}")
    print()
    print()
    weights += dw
    return weights


def calculate_exp_reward(reward_history, window):
    window = 10
    return sum(reward_history[-window:]) / window
    

inputs = np.identity(2)
weights = np.zeros(2)

reward_ratio = (0.2, 0.8)
reward_in_arena = np.zeros(2)

prob_outs, ys, odors, reward_history = [], [], [], []

for i in range(20):
    # sample binary decision with prob_output probability
    decision = 0
    ys.append([])
    odors.append([])
    prob_outs.append([])
    # update
    sampled_reward = np.zeros(2)
    if np.random.rand() < reward_ratio[0]:
        sampled_reward[0] = 1
    if np.random.rand() < reward_ratio[1]:
        sampled_reward[1] = 1

    reward_in_arena = np.logical_or(reward_in_arena, sampled_reward)

    # begin trial
    while decision == 0:
        odor = np.random.choice([0, 1])
        x = inputs[odor]
        prob_out = sigmoid(np.matmul(x, weights))
        decision = np.random.choice([0,1], p=(1-prob_out, prob_out))
        prob_outs[-1].append(prob_out)
        ys[-1].append(decision)
        odors[-1].append(odor)

        if decision == 1:
            reward = reward_in_arena[odor]
            print(f"reward in arena before collection: {reward_in_arena}")
            print("odor choice: ", odor)
            print("reward: ", reward)
            reward_in_arena[odor] = 0
            weights = update_weights(x, reward, reward_history, weights)
            reward_history.append(reward)
            break

print("prob_outs: ", prob_outs)
print() 
print("ys: ")
pprint(ys)
print("odors: ")
pprint(odors)
print()
print("average odor choice: ", np.mean([odor[-1] for odor in odors]))

