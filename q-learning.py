# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import random

import gym
import numpy as np

import caveman_world  # registers 'ewall/CavemanWorld-v1' env
import frozen_lake_mod  # registers 'ewall/FrozenLakeModified-v1' env
from vi_and_pi import evaluate_policy, timing


MAX_ITER = 10 ** 3
SEED = 1


class QLearner(object):

	### Code Credit: this code is my own implementation from Fall 2018, but portions are built off this template:
	#   Title: Template for implementing QLearner
	#   Author: Tucker Balch
	#   Date: 2018
	#   Availability: only via Georgia Tech course CS7646

	def __init__(self,
	             num_states=100,
	             num_actions=4,
	             alpha=0.2,
	             gamma=0.9,
	             rar_start=1.0,
	             rar_decay=0.005,
	             verbose=False):
		"""
		Initialize QLearner object.
		:param num_states: Total number of possible states; integer.
		:param num_actions: Total number of possible actions; integer.
		:param alpha: Learning rate; float between 0.0 and 1.0.
		:param gamma: Discount rate; float between 0.0 and 1.0.
		:param rar_start: Initial random action rate; float between 0.0 and 1.0.
		:param rar_decay: Random action decay rate; float between 0.0 and 1.0.
		:param verbose: Enable debugging printouts.
		"""

		# sanity check
		inputs = {'alpha':alpha, 'gamma':gamma, 'rar_start':rar_start, 'rar_decay':rar_decay}
		for key in inputs:
			if inputs[key] < 0 or inputs[key] > 1:
				raise ValueError(key + " value must be between 0.0 and 1.0.")

		# save config
		self.num_actions = num_actions
		self.alpha = alpha
		self.gamma = gamma
		self.rar_start = rar_start
		self.rar_decay = rar_decay
		self.verbose = verbose

		# set the stage
		self.s = None  # previous state
		self.a = None  # previous action
		self.q = np.zeros((num_states, num_actions))  # Q table

		# exponential decay
		self.random_action_rate = rar_start
		self.t = 1  # counter runs over lifetime of the object, not reset with each episode

	def get_policy(self):
		""" Return current best policy """
		return self.q.argmax(axis=1)

	def get_v(self):
		""" Return current best V """
		return self.q.max(axis=1)

	def reset(self, s=0):
		""" Reset to initial state and clear counters"""
		#TODO clear counters?
		return self.query_and_set_state(s)

	def query_and_set_state(self, s):
		"""
		@summary: Update the state without updating the Q-table
		@param s: The new state
		@returns: The selected action
		"""

		action = np.argmax(self.q[s, :])

		if self.verbose:
			print("set: s =", s, "a =", action)

		self.s = s
		self.a = action

		return action

	@timing
	def run(self, env, max_iterations=MAX_ITER):
		""" TBD """

		q_variation, epoch_rewards = [], []

		action = self.reset()  # set the initial state and get first action
		for i in range(max_iterations):
			total_reward = 0
			state, reward, done, _ = env.step(action)
			total_reward += reward

			if done:
				epoch_rewards.append(total_reward)
				s = env.reset()
				self.reset(s)

			prev_q = self.q[self.s, self.a]  # for variation
			action = self.update_and_query(state, reward)

			q_var = abs(prev_q - self.q[self.s, self.a])
			q_variation.append(q_var)

			#TODO WHEN DO WE STOP?!?

		return self.get_policy(), i + 1, q_variation, epoch_rewards

	def update_and_query(self, s_prime, r):
		"""
		@summary: Update the Q table and return an action
		@param s_prime: The new state
		@param r: The reward value
		@returns: The selected action
		"""

		if self.verbose:
			print("\nquery: s' =", s_prime, "r =", r)

		# calculate Q value
		prev_q = self.q[self.s, self.a]
		future_q =  r + self.gamma * self.q[s_prime, np.argmax(self.q[s_prime, :])]
		self.q[self.s, self.a] = (1 - self.alpha) * prev_q + (self.alpha * future_q)

		if self.verbose:
			print("   prev_q:", prev_q, "future_q:", future_q, "q_value:", self.q[self.s, self.a])

		# decide if action will be random
		if random.random() <= self.random_action_rate:
			# yes, we're returning a random action
			action = random.randint(0, self.num_actions - 1)
		else:
			# query Q table for the action
			action = np.argmax(self.q[s_prime, :])
		if self.verbose:
			print("   a =", action)

		# decay the random action rate for next time
		self.random_action_rate = 1.0 * np.exp(-self.rar_start * self.rar_decay * self.t)
		self.t += 1
		if self.random_action_rate < 0:
			self.random_action_rate = 0.0
		if self.verbose:
			print("   rar=", self.random_action_rate, "at t=", self.t)

		# store S' and A' for next time
		self.s = s_prime
		self.a = action

		return action


def run_and_evaluate(env_name):
	env = gym.make(env_name)
	env.seed(SEED)
	s = env.reset()
	print('== {} =='.format(env_name))

	#TODO loop thru different settings for plotting

	# build Q-learner
	ql = QLearner(num_states=env.observation_space.n,
	             num_actions=env.action_space.n,
	             alpha=0.9,
	             gamma=0.9,
	             rar_start=1.0,
	             rar_decay=0.005)
	ql.reset(s)

	# run learner
	policy, iters, q_variation, epoch_rewards = ql.run(env)

	print('== Q-Learning ==')
	print('Iterations:', iters)
	print('Variation curve:', q_variation)
	print('Rewards curve:', epoch_rewards, '\n')

	print('== QL Policy ==')
	print(policy)
	env.print_policy(policy)

	score = evaluate_policy(env, policy)
	print('Average total reward', score)

	return policy

if __name__ == "__main__":

	# seed pseudo-RNG for reproducibility
	random.seed(SEED)
	np.random.seed(SEED)

	# run Frozen Lake Modified (large grid problem)
	#run_and_evaluate('ewall/FrozenLakeModified-v1')

	# # run Caveman's World (simple problem)
	run_and_evaluate('ewall/CavemanWorld-v1')