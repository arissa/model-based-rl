import gym
import math
import random
import numpy as np
import pickle
from dataset import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# Hyperparameters
policy_hidden_size = 32
model_hidden_size = 1024
state_size = 10  # 8 obs states + 2 action states

learning_rate = 0.001
gamma = 0.99  # discount factor for reward
num_policy_opt = 100
max_timestep = 100
###################
# Define Networks #
###################


class Model(nn.Module):
    def __init__(self, observation_size, state_size, hidden_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, observation_size)

    def forward(self, state):
        hidden = F.relu(self.linear1(state))
        predicted_observation = self.linear2(hidden)

        return predicted_observation


class Policy(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(observation_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = F.relu(self.linear1(state))
        action = F.tanh(self.linear2(state))
        return action


# def train_model(model, model_optimizer, obs_buffer, action_buffer):
#     print("training model")
#     # Using previous states we will get predicted values for already known real parameters
#     # previous_actions = torch.abs(action_buffer[:-1] - 1)
#
#     # make sure action_buffer is the right shape and is being indexed correctly
#     previous_actions = action_buffer[:-1]
#     previous_states = torch.cat([obs_buffer[:-1, :], previous_actions], 1)
#
#     # Real parameters
#     true_next_observation = Variable(obs_buffer[1:, :], requires_grad=False)
#
#     # Get predictions
#     # we actually want to predict the change in observation, not the next observation
#     predicted_next_observation = model(Variable(previous_states, requires_grad=False))
#
#     # Calculate losses
#     observation_loss = (true_next_observation - predicted_next_observation).pow(2)
#     model_loss = torch.mean(observation_loss)
#
#     # model_optimizer.zero_grad()
#     model_loss.backward()
#     model_optimizer.step()

def train_model(model, model_optimizer, prev_states, next_obs):
    prev_states = torch.from_numpy(prev_states).type(torch.FloatTensor)
    next_obs = torch.from_numpy(next_obs).type(torch.FloatTensor)
    predicted_next_obs = model(Variable(prev_states, requires_grad=False))
    # print(next_obs)
    # print(predicted_next_obs)
    next_obs = Variable(next_obs)
    obs_loss = (next_obs - predicted_next_obs).pow(2)
    model_loss = torch.mean(obs_loss)
    model_loss.backward()
    model_optimizer.step()



def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = torch.zeros(r.size())
    running_add = 0
    for t in reversed(range(0, r.numel())):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def cost(obs, a, next_obs):
    # TODO: need to add cases for first and last
    cost_coeff = 1e-4
    return torch.mul(torch.sum(torch.pow(a, 2)), cost_coeff)


# input to policy network: obs
# output from policy network: action
# loss: reward based on the action outputted from policy network?
def train_policy(observation, model, policy, policy_optimizer):
    # print("training policy")
    loss = 0.
    for t in range(num_policy_opt):

        action = policy(Variable(observation))  # forward pass to get actions
        action_tensor = torch.from_numpy(action.data.numpy()).type(torch.FloatTensor)

        # state = torch.cat((obs_buffer[:-1,:], action), 1)
        state = torch.cat((observation, action_tensor), 1)

        # print(state)

        # in the model, simulate the trajectories and compute the summed discounted reward
        next_observation = model(Variable(state))#, requires_grad=False))
        next_observation = torch.from_numpy(next_observation.data.numpy()).type(torch.FloatTensor)

        loss  += cost(observation, action, next_observation)

        observation = next_observation

    # policy_optimizer.zero_grad()
    policy_loss = loss
    # print(policy_loss)
    policy_loss.backward()
    policy_optimizer.step()


# collect samples from the real environment, using the policy
def collect_samples(env, policy, batch_size):
    observations = []
    actions = []
    rewards = []

    # max_eps_reward = -np.inf
    # min_eps_reward = np.inf
    # avg_eps_reward = 0.0

    total_timesteps = 0
    while total_timesteps < batch_size:
        obs_buffer = []
        action_buffer = []
        reward_buffer = []

        obs = env.reset()
        obs_buffer.append(obs)
        episode_reward = 0.0
        for t in range(max_timestep):
            obs = torch.from_numpy(obs).type(torch.FloatTensor).view(1, env.observation_space.shape[0])
            action = policy(Variable(obs))
            action = action.data.numpy()
            obs, reward, done, info = env.step(action)
            obs_buffer.append(obs)
            # this says action[0] in the other code. why?
            action_buffer.append(action[0])
            reward_buffer.append(reward)
            episode_reward += reward
            total_timesteps += 1
            if done:
                break
        observations.append(obs_buffer)
        actions.append(action_buffer)
        rewards.append(reward_buffer)

    return observations, actions, rewards


def model_based_rl():
    env = gym.make('Swimmer-v2')
    #########################
    # Initialize Optimizers #
    #########################

    # initialize policy
    policy = Policy(
        observation_size=env.observation_space.shape[0],
        action_size=2,
        hidden_size=policy_hidden_size
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    policy_optimizer.zero_grad()

    # initialize model
    model = Model(
        observation_size=env.observation_space.shape[0],
        state_size=state_size,
        hidden_size=model_hidden_size
    )
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_optimizer.zero_grad()

    while True:
        # # initialize empty dataset D
        # observation_history = []
        # reward_history = []
        # action_history = []
        # status_history = []
        #
        # # collect samples from environment using policy and add them to D
        # for i_episode in range(30):
        #     observation = env.reset()
        #
        #     episode_reward = 0.
        #     for t in range(max_timestep):
        #         if isinstance(observation, np.ndarray):
        #             observation = torch.from_numpy(observation).type(torch.FloatTensor).view(1,
        #                                                                                      env.observation_space.shape[
        #                                                                                          0])
        #
        #         observation_history.append(observation)
        #
        #         # env.render()
        #         action = policy(Variable(observation, requires_grad=False))
        #         action = action.data.numpy()
        #
        #         # action = env.action_space.sample()
        #
        #         action_history.append(action)
        #         if (action != action).any():
        #             print("Action is NaN")
        #         observation, reward, done, info = env.step(action)
        #         reward_history.append(reward)
        #         status_history.append(done * 1)
        #
        #         episode_reward += reward
        #         # print('Reward %f.' % (reward))
        #
        #         if done or t == 99:
        #             print("Episode finished after {} timesteps".format(t + 1))
        #             print('Reward %f.' % (episode_reward))
        #             break
        # observation_buffer = torch.cat(observation_history)
        # action_buffer = torch.from_numpy(np.vstack(action_history)).type(torch.FloatTensor)
        #
        # print("obs_buffer")
        # print(observation_buffer.shape)
        # print("action_buffer")
        # print(action_buffer.shape)

        D = Dataset()
        observations, actions, rewards = collect_samples(env, policy, 1000)
        prev_states = []
        next_states = []
        for i, observation_traj in enumerate(observations):
            action_traj = actions[i]
            for t in range(len(observation_traj) - 1):
                # print(observation_traj[t])
                # print(action_traj[t])
                prev_states.append(np.concatenate([observation_traj[t], action_traj[t]]))
                next_states.append(observation_traj[t+1])
        prev_states = np.array(prev_states)
        next_states = np.array(next_states)
        D.add_data(np.array(prev_states), np.array(next_states))

        # print reward of 1 trajectory:
        print("reward %f." % (np.sum(rewards[0])))

        # print("prev states")
        # print(prev_states.shape)
        # print("next states")
        # print(next_states.shape)
        # # all_prev_states = []
        # # all_next_states = []
        #
        # model optimization
        # # TODO: pretty sure this should be batched
        for j in range(1000):
            prev_states, next_states = D.get_next_batch(32)
            train_model(model, model_optimizer, prev_states, next_states)

        # policy optimization
        for i_episode in range(30):
            # print("EPISODE %d" % (i_episode))
            observation = env.reset()

            episode_reward = 0.
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).type(torch.FloatTensor).view(1, env.observation_space.shape[0])
            train_policy(observation, model, policy, policy_optimizer)


def main():

    model_based_rl()


main()
