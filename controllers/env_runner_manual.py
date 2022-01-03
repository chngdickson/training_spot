import gym
from EnvRunner import EnvRunner

env = gym.make('BipedalWalker-v3')

runner = EnvRunner(env, terminalReward=-40.0, rewardScale=1.0) # Cart-Pole environment always returns a reward of 1, so use a custom reward function: -1 if episode ends, 0 otherwise

for episode in range(10000):
    env.reset()

    # Timesteps
    for t in range(500):
        done, _ = runner.act() # Step the environment and agent

        if done:
            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))
            
            break