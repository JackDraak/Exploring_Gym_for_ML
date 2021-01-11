#
# Use Q-learning to train against the Frozen Lake game
#

import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# Import a prepared gym environment for Frozen Lake
env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# The q_table, initialized as a matrix of zeros
q_table = np.zeros((state_space_size, action_space_size))
print("** Initialized Q-table **\n\n", q_table, "\n")

# Juicy variables to tinker with to see if you can get better training
num_episodes = 6000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 0.75
min_exploration_rate = 0.003
exploration_decay_rate = 0.001

# Q-Learning Algorithm
rewards_all_episodes = []
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        # Make an action in the environment
        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s, a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (
                reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
        -exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_ep = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("*** Average reward per thousand episodes ***\n")
for r in rewards_per_thousand_ep:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Print updated Q-table
print("\n*** Updated Q-table ***\n")
print(q_table)

# Watch the trained agent, the q_table, in action:
for episode in range(3):
    state = env.reset()
    done = False
    print(" *** Episode: ", episode + 1, " ***\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("*** Goal Reached ***")
                time.sleep(2)
            else:
                print("--- Fell Through a Hole ---")
                time.sleep(2)
            clear_output(wait=True)
            break

        state = new_state

    env.close()
