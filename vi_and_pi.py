# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import timeit

import gym
import numpy as np

import caveman_world
import frozen_lake_mod


MAX_ITER = 1000
SEED = 1


### Code Credit -- multiple functions on this page were modified from:
#   Title: OpenAI Gym Solutions
#   Author: Allan Brayes
#   Date: 2017-04-22
#   Code version: commit 20481fc1e1a2fdfcba5b0ce27927e5e29a122fc4
#   Availability: https://github.com/allanbreyes/gym-solutions/blob/master/analysis/mdp.py

def timing(f):
	def wrap(*args):
		start_time = timeit.default_timer()
		ret = f(*args)
		elapsed = timeit.default_timer() - start_time
		print('%s function took %0.5f seconds' % (f.__name__, elapsed))
		return ret

	return wrap


def evaluate_rewards_and_transitions(problem, mutate=False):
	# Enumerate state and action space sizes
	num_states = problem.observation_space.n
	num_actions = problem.action_space.n

	# Intiailize T and R matrices
	R = np.zeros((num_states, num_actions, num_states))
	T = np.zeros((num_states, num_actions, num_states))

	# Iterate over states, actions, and transitions
	for state in range(num_states):
		for action in range(num_actions):
			for transition in problem.env.P[state][action]:
				probability, next_state, reward, done = transition
				R[state, action, next_state] = reward
				T[state, action, next_state] = probability

			# Normalize T across state + action axes
			T[state, action, :] /= np.sum(T[state, action, :])

	# Conditionally mutate and return
	if mutate:
		problem.env.R = R
		problem.env.T = T
	return R, T


@timing
def value_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10 ** 6, delta=10 ** -3):
	""" Runs Value Iteration on a gym problem """

	value_fn = np.zeros(problem.observation_space.n)
	if R is None or T is None:
		R, T = evaluate_rewards_and_transitions(problem)

	for i in range(max_iterations):
		previous_value_fn = value_fn.copy()
		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		value_fn = np.max(Q, axis=1)

		if np.max(np.abs(value_fn - previous_value_fn)) < delta:
			break

	# Get and return optimal policy
	policy = np.argmax(Q, axis=1)
	return policy, i + 1


@timing
def policy_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10 ** 6, delta=10 ** -3):
	""" Runs Policy Iteration on a gym problem """

	def encode_policy(policy, shape):
		""" One-hot encodes a policy """
		encoded_policy = np.zeros(shape)
		encoded_policy[np.arange(shape[0]), policy] = 1
		return encoded_policy

	num_spaces = problem.observation_space.n
	num_actions = problem.action_space.n

	# Initialize with a random policy and initial value function
	policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
	value_fn = np.zeros(num_spaces)

	# Get transitions and rewards
	if R is None or T is None:
		R, T = evaluate_rewards_and_transitions(problem)

	# Iterate and improve policies
	for i in range(max_iterations):
		previous_policy = policy.copy()

		for j in range(max_iterations):
			previous_value_fn = value_fn.copy()
			Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
			value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

			if np.max(np.abs(previous_value_fn - value_fn)) < delta:
				break

		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		policy = np.argmax(Q, axis=1)

		if np.array_equal(policy, previous_policy):
			break

	# Return optimal policy
	return policy, i + 1


def run_discrete(environment_name):
	problem = gym.make(environment_name)
	problem.seed(SEED)
	print('== {} =='.format(environment_name))
	print('Actions:', problem.env.action_space.n)
	print('States:', problem.env.observation_space.n)
	problem.print_grid()

	print('== Value Iteration ==')
	value_policy, iters = value_iteration(problem)
	print('Iterations:', iters, '\n')

	print('== VI Policy ==')
	problem.print_policy(value_policy)

	print('== Policy Iteration ==')
	policy, iters = policy_iteration(problem)
	print('Iterations:', iters, '\n')

	print('== PI Policy ==')
	problem.print_policy(policy)

	diff = sum([abs(x - y) for x, y in zip(policy.flatten(), value_policy.flatten())])
	print('Discrepancy:', diff, '\n')

	return policy


if __name__ == "__main__":

	# run Frozen Lake Modified (large grid problem)
	run_discrete('ewall/FrozenLakeModified-v1')

	# # run Caveman's World (simple problem)
	run_discrete('ewall/CavemanWorld-v1')
