import gym
from gym import wrappers

env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/monitor')
observation = env.reset()
for _ in range(1000):
	env.render()
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)

	if done:
		env.reset()

env.close()