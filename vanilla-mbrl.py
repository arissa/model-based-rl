
import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

class Model(nn.Module):
	def __init__(self, state_space, hidden_size, reward_space, observation_space):
		super(Model, self).__init__()
		self.linear1 = nn.Linear(state_space, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)

		# Predict observation, reward and status separately
		self.observation_layer = nn.Linear(hidden_size, observation_space)
		self.reward_layer = nn.Linear(hidden_size, reward_space)
		self.status_layer = nn.Linear(hidden_size, 1)
	
	def forward(self, state):
		state = F.relu(self.linear1(state))
		hidden = F.relu(self.linear2(state))
		predicted_observation = self.observation_layer(hidden)
		predicted_reward = self.reward_layer(hidden)
		predicted_status = F.sigmoid(self.status_layer(hidden))
		
		return predicted_observation, predicted_reward, predicted_status


class Policy(nn.Module):
	def __init__(self, observation_space, action_space, hidden_size):
		super(Policy, self).__init__()
		self.linear1 = nn.Linear(observation_space, hidden_size)
		self.linear2 = nn.Linear(hidden_size, action_space)

	def forward(self, state):
		state = F.tanh(self.linear1(state))
		action = F.tanh(self.linear2(state))
		return action

def train_model(model, model_optimizer, obs_buffer, action_buffer, rewards_buffer, status_buffer):
	print("training model")
	# Using previous states we will get predicted values for already known real parameters
	# previous_actions = torch.abs(action_buffer[:-1] - 1)
	previous_actions = action_buffer[:-1]
	previous_states = torch.cat([obs_buffer[:-1,:], previous_actions], 1)

	# Real parameters
	true_observations = Variable(obs_buffer[1:,:], requires_grad=False)
	true_rewards = Variable(rewards_buffer[1:,:], requires_grad=False)
	true_status = Variable(status_buffer[1:,:], requires_grad=False)
	
	# Get predictions
	# we actually want to predict the change in observation, not the next observation
	predicted_observation, predicted_reward, predicted_done = model(Variable(previous_states, requires_grad=False))

	# Calculate losses
	observation_loss = (true_observations - predicted_observation).pow(2)
	reward_loss = (true_rewards - predicted_reward).pow(2)
	done_loss = torch.mul(predicted_done, true_status) + torch.mul(1 - predicted_done, 1 - true_status)
	done_loss = -torch.log(done_loss)

	model_loss = torch.mean(observation_loss)
	# model_loss = torch.mean(
		# observation_loss + done_loss.expand_as(observation_loss) + reward_loss.expand_as(observation_loss)
	# )
	# print("loss: %f " % (model_loss))
	# Update
	model_optimizer.zero_grad()
	model_loss.backward()
	model_optimizer.step()

# Hyperparameters
policy_hidden_size = 32
model_hidden_size = 1024
state_space = 10

learning_rate = 0.001
gamma = 0.99  # discount factor for reward

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = torch.zeros(r.size())
	running_add = 0
	for t in reversed(range(0, r.numel())):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


#input to policy network: obs
#output from policy network: action
#loss: reward based on the action outputted from policy network? 
def train_policy(obs_buffer, model, policy, policy_optimizer):
	action = policy(Variable(obs_buffer))  #forward pass to get actions
	action = torch.from_numpy(action.data.numpy()).type(torch.FloatTensor)
	# print(obs_buffer)
	# print(action)
	# state = torch.FloatTensor([[obs_buffer, action]])
	# action = torch.FloatTensor(action)
	# state = torch.cat((obs_buffer[:-1,:], action), 1)
	state = torch.cat((obs_buffer, action), 1)

	# print(state)

	# this only gives the reward at next timestep though
	_, reward, _ = model(Variable(state, requires_grad=False))

	loss = -torch.sum(reward)
	# print(loss)
	policy_optimizer.zero_grad()

	loss.backward()
	policy_optimizer.step()




def model_based_rl():
	env = gym.make('Swimmer-v2')

	# initialize policy
	policy = Policy(
		observation_space=env.observation_space.shape[0],  # State Space is a Box(4,)
		action_space=2,
		hidden_size=policy_hidden_size
	)
	policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
	policy_optimizer.zero_grad()

	# initialize model
	model = Model(
		observation_space=env.observation_space.shape[0],  # State Space is a Box(4,)
		hidden_size=model_hidden_size,
		reward_space = 1,
		state_space = state_space
	)
	model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	model_optimizer.zero_grad()


	while True:
		#initialize empty dataset D
		observation_history = []
		reward_history = []
		action_history = []
		status_history = [] 

		# collect samples from environment using policy and add them to D
		for i_episode in range(30):
			observation = env.reset()
   
			episode_reward = 0.
			for t in range(100):
				if isinstance(observation, np.ndarray):
					observation = torch.from_numpy(observation).type(torch.FloatTensor).view(1, env.observation_space.shape[0])

				observation_history.append(observation)

				# env.render()
				# this should be from the policy, not random
				action = policy(Variable(observation, requires_grad=False)) 
				action = action.data.numpy()

				# action = env.action_space.sample()

				action_history.append(action)

				observation, reward, done, info = env.step(action)
				reward_history.append(reward)
				status_history.append(done*1)

				episode_reward += reward
				# print('Reward %f.' % (reward))

				if done or t ==99:
					print("Episode finished after {} timesteps".format(t+1))
					print('Reward %f.' % (episode_reward))
					break
		observation_buffer = torch.cat(observation_history)
		action_buffer = torch.from_numpy(np.vstack(action_history)).type(torch.FloatTensor)
		rewards_buffer = torch.from_numpy(np.vstack(reward_history)).type(torch.FloatTensor)
		status_buffer = torch.from_numpy(np.vstack(status_history)).type(torch.FloatTensor)
				
		train_model(model, model_optimizer, observation_buffer, action_buffer, rewards_buffer, status_buffer)



		for i_episode in range(30):
			# print("EPISODE %d" % (i_episode))
			observation = env.reset()
			observation_samples = []
   
			episode_reward = 0.
			for t in range(100):
				if isinstance(observation, np.ndarray):
					observation = torch.from_numpy(observation).type(torch.FloatTensor).view(1, env.observation_space.shape[0])

				observation_samples.append(observation)
				action = policy(Variable(observation, requires_grad=False)) # should it be observation_history?
				action = action.data.numpy()

				observation, reward, done, info = env.step(action)

				episode_reward += reward

				if done or t == 99:
					print("Episode finished after {} timesteps".format(t+1))
					print('Reward %f.' % (episode_reward))
					observation_buffer = torch.cat(observation_samples)
					train_policy(observation_buffer, model, policy, policy_optimizer)
					break


def main():
	model_based_rl()
main()