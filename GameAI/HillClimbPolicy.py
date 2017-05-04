import gym
import numpy as np

# Hill climbing initialize weights randomly, utilize memory to save food weigths.
def run_episode(env, parameters):

	observation = env.reset()
	total_reward = 0

	for _ in range(200):
		env.render()
		# initialize random weights
		action = 0 if np.matmul(parameters, observation)<0 else 1
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

def train(submit):
	env = gym.make('CartPole-v0')

	episodes_per_update = 5
	noise_scaling = 0.1
	parameters = np.random.rand(4) * 2 - 1
	best_reward = 0

	#2000 episodes
	for _ in range(2000):
		new_params = parameters + (np.random.rand(4) *2 -1) * noise_scaling
		reward = run_episode(env, new_params)
		print("Reward %d best %d" %(reward, best_reward))

		if reward > best_reward:
			best_reward = reward
			parameters = new_params
			if reward == 200:
				break

run = train(submit=False)
print(run)