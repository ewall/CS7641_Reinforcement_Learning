# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import random
import timeit

import gym
import numpy as np

import caveman_world  # registers 'ewall/CavemanWorld-v1' env
import frozen_lake_mod  # registers 'ewall/FrozenLakeModified-v1' env


MAX_ITER = 1000
SEED = 1


### Code Credit -- multiple functions on this page were modified from:
#   Title: OpenAI Gym Solutions
#   Author: Allan Brayes
#   Date: 2017-04-22
#   Code version: commit 20481fc1e1a2fdfcba5b0ce27927e5e29a122fc4
#   Availability: https://github.com/allanbreyes/gym-solutions/blob/master/analysis/mdp.py

### Code Credit -- other functions on this page were modified from:
#   Title: Deep Reinforcement Learning Demysitifed (Episode 2) — Policy Iteration, Value Iteration and Q-learning
#   Author: Moustafa Alzantot (malzantot@ucla.edu)
#   Date: 2017-07-09
#   Code version: https://gist.github.com/malzantot/ed173b66e76a05e9c8eeec60dd476948
#   Availability: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

def timing(f):
	""" Simple decorator to time a function's execution """
	def wrap(*args):
		start_time = timeit.default_timer()
		ret = f(*args)
		elapsed = timeit.default_timer() - start_time
		print('%s function took %0.5f seconds' % (f.__name__, elapsed))
		return ret

	return wrap


def get_r_and_t(problem):
	""" Generate R and T matrices from the problem's transitions """

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

	# save into the env for re-use
	problem.env.R = R
	problem.env.T = T

	return R, T


@timing
def value_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10 ** 6, delta=10 ** -3):
	""" Runs Value Iteration on a gym problem """

	value_fn = np.zeros(problem.observation_space.n)
	errors = []

	# get transitions and rewards
	if R is None or T is None:
		if hasattr(problem.env, 'R') and hasattr(problem.env, 'T'):
			R, T = problem.env.R, problem.env.T
		else:
			R, T = get_r_and_t(problem)

	# iterate and improve value function
	for i in range(max_iterations):
		previous_value_fn = value_fn.copy()
		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		value_fn = np.max(Q, axis=1)

		err = np.max(np.abs(value_fn - previous_value_fn))
		errors.append(err)
		if err < delta:
			break

	# get and return optimal policy
	policy = np.argmax(Q, axis=1)
	return policy, i + 1, errors


@timing
def policy_iteration(problem, R=None, T=None, gamma=0.9, max_iterations=10 ** 6, delta=10 ** -3):
	""" Runs Policy Iteration on a gym problem """

	def encode_policy(policy, shape):
		""" one-hot encode a policy """
		encoded_policy = np.zeros(shape)
		encoded_policy[np.arange(shape[0]), policy] = 1
		return encoded_policy

	num_spaces = problem.observation_space.n
	num_actions = problem.action_space.n
	errors = []

	# initialize with a random policy and initial value function
	policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
	value_fn = np.zeros(num_spaces)

	# get transitions and rewards
	if R is None or T is None:
		if hasattr(problem.env, 'R') and hasattr(problem.env, 'T'):
			R, T = problem.env.R, problem.env.T
		else:
			R, T = get_r_and_t(problem)

	# iterate and improve policies
	for i in range(max_iterations):
		previous_policy = policy.copy()

		for j in range(max_iterations):
			previous_value_fn = value_fn.copy()
			Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
			value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

			err = np.max(np.abs(previous_value_fn - value_fn))
			errors.append(err)
			if err < delta:
				break

		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		policy = np.argmax(Q, axis=1)

		if np.array_equal(policy, previous_policy):
			break

	# return optimal policy
	return policy, i + 1, errors


def run_and_evaluate(environment_name):
	problem = gym.make(environment_name)
	problem.seed(SEED)
	print('== {} =='.format(environment_name))
	print('Actions:', problem.env.action_space.n)
	print('States:', problem.env.observation_space.n)
	problem.print_grid()

	print('== Value Iteration ==')
	vi_policy, iters, errs = value_iteration(problem)
	print('Iterations:', iters)
	print('Error curve:', errs, '\n')

	print('== VI Policy ==')
	vi_score = evaluate_policy(problem, vi_policy)
	print('Average total reward', vi_score)
	problem.print_policy(vi_policy)

	print('== Policy Iteration ==')
	pi_policy, iters, errs = policy_iteration(problem)
	print('Iterations:', iters)
	print('Error curve:', errs, '\n')

	print('== PI Policy ==')
	pi_score = evaluate_policy(problem, pi_policy)
	print('Average total reward', pi_score)
	problem.print_policy(pi_policy)

	diff = (vi_policy != pi_policy).flatten().sum()
	print('Discrepancy:', diff)
	if diff > 0:
		if vi_score > pi_score:
			print('Best score: VI')
		elif pi_score > vi_score:
			print('Best score: PI')
		else:
			print('Tied score')
	print()

	return pi_policy


def run_episode(env, policy, gamma=1.0, render=False):
	""" Runs a single episode on the given env and policy, and return the total reward """
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done , _ = env.step(policy[obs])
		total_reward += (gamma ** step_idx * reward)
		step_idx += 1
		if done:
			break
	return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=1000):
	""" Run a policy multiple times and return the average total reward """
	scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)


if __name__ == "__main__":

	# seed pseudo-RNG for reproducibility
	random.seed(SEED)
	np.random.seed(SEED)

	# run Frozen Lake Modified (large grid problem)
	run_and_evaluate('ewall/FrozenLakeModified-v1')

	# # run Caveman's World (simple problem)
	run_and_evaluate('ewall/CavemanWorld-v1')
