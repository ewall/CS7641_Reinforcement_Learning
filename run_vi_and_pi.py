# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import random
import timeit
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.envs import registration
import env_caveman_world  # registers 'ewall/CavemanWorld-v1' env
import env_frozen_lake_mod  # registers 'ewall/FrozenLakeModified-v1' & v2 (alternate reward) envs


MAX_ITER = 10 ** 3
SEED = 1
SHOW_PLOTS = False

### Code Credit -- multiple functions on this page were modified from:
#   Title: OpenAI Gym Solutions
#   Author: Allan Reyes
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
	def wrap(*args, **kwargs):
		start_time = timeit.default_timer()
		ret = f(*args, **kwargs)
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
def value_iteration(problem, gamma=0.9, delta=10 ** -3, max_iterations=MAX_ITER, R=None, T=None):
	""" Runs Value Iteration on a gym problem """

	value_fn = np.zeros(problem.observation_space.n)
	errors = []
	optimal_achieved = False

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

		# check error
		err = np.max(np.abs(value_fn - previous_value_fn))
		errors.append(err)
		if err < delta:
			print("Error %0.5f below delta on iteration %d" % (err, i + 1))
			break

		# check if optimal policy already achieved
		if hasattr(problem.env, 'optimal_policy') and optimal_achieved == False:
			current_policy = np.argmax(Q, axis=1)
			diff = diff_policies(current_policy, problem.optimal_policy)
			if diff == 0:
				optimal_achieved = True
				print("Optimal policy found on iteration", str(i + 1))

	# get and return optimal policy
	policy = np.argmax(Q, axis=1)
	return policy, i + 1, errors, None


@timing
def policy_iteration(problem, gamma=0.9, delta=10 ** -3, max_iterations=MAX_ITER, R=None, T=None):
	""" Runs Policy Iteration on a gym problem """

	def encode_policy(policy, shape):
		""" one-hot encode a policy """
		encoded_policy = np.zeros(shape)
		encoded_policy[np.arange(shape[0]), policy] = 1
		return encoded_policy

	num_spaces = problem.observation_space.n
	num_actions = problem.action_space.n
	errors = []
	steps = 1
	optimal_achieved, delta_achieved = False, False

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

			# check error
			err = np.max(np.abs(previous_value_fn - value_fn))
			errors.append(err)
			if err < delta:
				if delta_achieved == False:
					print("Error %0.5f below delta on iteration %d and step %d" % (err, i + 1, steps))
					delta_achieved = True
				steps += j
				break

			# check if optimal policy already achieved
			if hasattr(problem.env, 'optimal_policy') and optimal_achieved == False:
				current_policy = np.argmax(Q, axis=1)
				diff = diff_policies(current_policy, problem.optimal_policy)
				if diff == 0:
					optimal_achieved = True
					print("Optimal policy found on iteration %d and step %d" % (i + 1, j + 1))

		Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
		policy = np.argmax(Q, axis=1)

		if np.array_equal(policy, previous_policy):
			break

	# return optimal policy
	return policy, i + 1, errors, steps


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
	return total_reward, step_idx


def evaluate_policy(env, policy, gamma=1.0, n=1000):
	""" Run a policy multiple times and return the average total reward and average steps taken """
	scores, steps = [], []
	for _ in range(n):
		score, step = run_episode(env, policy, gamma, False)
		scores.append(score)
		steps.append(step)
	return scores, steps


def diff_policies(policy1, policy2):
	""" Count differences between two numpy arrays """
	return (policy1 != policy2).flatten().sum()


def run_and_evaluate(environment_name,
                     env_nickname="MDP",
                     print_grids=True,
                     gamma=0.9,
                     delta=10 ** -3,
                     max_iterations=10 ** 3,
                     plot=True):
	problem = gym.make(environment_name)
	problem.seed(SEED)
	print('== {} =='.format(environment_name))
	print('Actions:', problem.env.action_space.n)
	print('States:', problem.env.observation_space.n)
	if print_grids:
		problem.print_grid()
	print()

	print('== Value Iteration ==')
	print('gamma:', gamma, 'delta:', delta, 'max_iterations:', max_iterations)
	vi_policy, vi_iters, vi_errs, _ = value_iteration(problem, gamma, delta, max_iterations)
	print('Iterations:', vi_iters)
	if not plot:
		print('Error curve:', vi_errs)
	print()

	print('== VI Policy ==')
	if print_grids:
		problem.print_policy(vi_policy)
	vi_scores, vi_actions = evaluate_policy(problem, vi_policy)
	print('Average total reward:', np.mean(vi_scores), 'max reward:', np.max(vi_scores))
	print('Average actions taken:', np.mean(vi_actions), 'max actions:', np.max(vi_actions), '\n')

	print('== Policy Iteration ==')
	print('gamma:', gamma, 'delta:', delta, 'max_iterations:', max_iterations)
	pi_policy, pi_iters, pi_errs, pi_backups = policy_iteration(problem, gamma, delta, max_iterations)
	print('Iterations:', pi_iters)
	print('Backups:', pi_backups)
	if not plot:
		print('Error curve:', pi_errs)
	print()

	print('== PI Policy ==')
	if print_grids:
		problem.print_policy(pi_policy)
	pi_scores, pi_actions = evaluate_policy(problem, pi_policy)
	print('Average total reward:', np.mean(pi_scores), 'max reward:', np.max(pi_scores))
	print('Average actions taken:', np.mean(pi_actions), 'max actions:', np.max(pi_actions), '\n')

	diff = diff_policies(vi_policy, pi_policy)
	print('Discrepancy:', diff, '\n')

	if plot:
		df_err = pd.DataFrame(list(zip(vi_errs, pi_errs)), columns=('VI', 'PI'))
		df_err.index.title = "iterations"
		ax = plt.gca()
		df_err.plot(kind='line', ax=ax)
		plt.xlabel('iterations')
		plt.ylabel('error (delta from previous utility value)')
		plt.title(env_nickname + ': Compare VI & PI error curves')
		plt.savefig('plots/VIPI_' + env_nickname.replace(' ', '') + '_error_curve.png', bbox_inches='tight')
		if SHOW_PLOTS:
			plt.show()
		plt.close()

	return pi_policy


def run_gamma_comparison(environment_name, env_nickname="MDP", gamma_list=(0.7, 0.8, 0.9), plot=True):
	assert gamma_list is not None, "must provide list of gammas to test"

	problem = gym.make(environment_name)
	problem.seed(SEED)
	vi_rewards, pi_rewards, pol_diffs  = [], [], []

	print('== Gamma Comparison: {} ==\n'.format(environment_name))

	for gamma in gamma_list:
		print('-- Gamma: {} --'.format(gamma))

		vi_policy, _, _, _ = value_iteration(problem, gamma=gamma)
		vi_scores, _ = evaluate_policy(problem, vi_policy)
		score = np.mean(vi_scores)
		vi_rewards.append(score)
		print('VI mean reward:', score)

		pi_policy, _, _, _ = policy_iteration(problem, gamma=gamma)
		pi_scores, _ = evaluate_policy(problem, pi_policy)
		score = np.mean(pi_scores)
		pi_rewards.append(score)
		print('PI mean reward:', score)

		diff = diff_policies(vi_policy, pi_policy)
		pol_diffs.append(diff)
		print('Discrepancy:', diff, '\n')

	if plot:
		# create dataframe
		df = pd.DataFrame(list(zip(vi_rewards, pi_rewards)), index=gamma_list, columns=('VI', 'PI'))
		df.index.title = "gamma"

		# create rewards plot
		ax = plt.gca()
		df.plot(kind='line', ax=ax)
		plt.xlabel('gamma values')
		plt.ylabel('mean reward')
		plt.title(env_nickname + ': Compare VI & PI across gamma values')
		plt.savefig('plots/VIPI_' + env_nickname.replace(' ', '') + '_gamma_rewards.png', bbox_inches='tight')
		if SHOW_PLOTS:
			plt.show()
		plt.close()

		# plot differences
		ax = plt.gca()
		df = pd.DataFrame(pol_diffs, index=gamma_list)
		df.plot(kind='line', ax=ax)
		ax.get_legend().remove()
		plt.xlabel('gamma values')
		plt.ylabel('differences between output policies')
		plt.title(env_nickname + ': Comparing VI & PI policies')
		plt.savefig('plots/VIPI_' + env_nickname.replace(' ', '') + '_gamma_diffs.png', bbox_inches='tight')
		if SHOW_PLOTS:
			plt.show()
		plt.close()

	return vi_rewards, pi_rewards, pol_diffs


def run_delta_comparison(environment_name, env_nickname="MDP", delta_list=(1.0, 0.1, 0.1), plot=True):
	assert delta_list is not None, "must provide list of deltas to test"

	problem = gym.make(environment_name)
	problem.seed(SEED)
	vi_rewards, pi_rewards, pol_diffs  = [], [], []

	print('== Delta Comparison: {} ==\n'.format(environment_name))

	for delta in delta_list:
		print('-- Delta: {} --'.format(delta))

		vi_policy, _, _, _ = value_iteration(problem, delta=delta)
		vi_scores, _ = evaluate_policy(problem, vi_policy)
		score = np.mean(vi_scores)
		vi_rewards.append(score)
		print('VI mean reward:', score)

		pi_policy, _, _, _ = policy_iteration(problem, delta=delta)
		pi_scores, _ = evaluate_policy(problem, pi_policy)
		score = np.mean(pi_scores)
		pi_rewards.append(score)
		print('PI mean reward:', score)

		diff = diff_policies(vi_policy, pi_policy)
		pol_diffs.append(diff)
		print('Discrepancy:', diff, '\n')

	if plot:
		# create dataframe
		df = pd.DataFrame(list(zip(vi_rewards, pi_rewards)), index=delta_list, columns=('VI', 'PI'))
		df.index.title = "delta"

		# create rewards plot
		ax = plt.gca()
		df.plot(kind='line', ax=ax)
		plt.xlabel('delta values')
		plt.ylabel('mean reward')
		plt.title(env_nickname + ': Compare VI & PI across delta values')
		plt.savefig('plots/VIPI_' + env_nickname.replace(' ', '') + '_delta_rewards.png', bbox_inches='tight')
		if SHOW_PLOTS:
			plt.show()
		plt.close()

		# plot differences
		ax = plt.gca()
		df = pd.DataFrame(pol_diffs, index=delta_list)
		df.plot(kind='line', ax=ax)
		ax.get_legend().remove()
		plt.xlabel('delta values')
		plt.ylabel('differences between output policies')
		plt.title(env_nickname + ': Comparing VI & PI policies')
		plt.savefig('plots/VIPI_' + env_nickname.replace(' ', '') + '_delta_diffs.png', bbox_inches='tight')
		if SHOW_PLOTS:
			plt.show()
		plt.close()

	return vi_rewards, pi_rewards, pol_diffs


if __name__ == "__main__":

	# seed pseudo-RNG for reproducibility
	random.seed(SEED)
	np.random.seed(SEED)

	### CAVEMAN WORLD ###

	# run Caveman's World (simple problem)
	run_and_evaluate('ewall/CavemanWorld-v1', env_nickname='Caveman World (delta=1.0)', delta=1.0)

	# run Caveman with a single iteration
	run_and_evaluate('ewall/CavemanWorld-v1', env_nickname='Caveman World (one iteration)', max_iterations=1, plot=False)

	# run Caveman comparing different gamma values
	gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95]
	vi_rewards, pi_rewards, pol_diffs = run_gamma_comparison('ewall/CavemanWorld-v1', 'Caveman World', gammas)

	### FROZEN LAKE ###

	# run Frozen Lake with Original Rewards, comparing different gamma values
	gammas = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]
	vi_rewards_orig, pi_rewards_orig, pol_diffs_orig = run_gamma_comparison('ewall/FrozenLakeModified-v1',
	                                                                        'Frozen Lake (Original Rewards)',
	                                                                        gammas)

	# run Frozen Lake with Alternate Rewards, comparing different gamma values
	vi_rewards_alt, pi_rewards_alt, pol_diff_alt = run_gamma_comparison('ewall/FrozenLakeModified-v2',
	                                                                    'Frozen Lake (Alternate Rewards)',
	                                                                    gammas)

	# run Frozen Lake (Alt), comparing different delta values
	deltas = [1.0, 0.1, 0.01, 0.001, 0.0001]
	vi_rewards_d, pi_rewards_d, pol_diffs_d = run_delta_comparison('ewall/FrozenLakeModified-v2', 'Frozen Lake', deltas)

	# "best" Frozen Lake (Orig)
	run_and_evaluate('ewall/FrozenLakeModified-v1', 'Frozen Lake (Original Reward)', gamma=0.9995, delta=0.0001)

	# "best" Frozen Lake (Alt)
	run_and_evaluate('ewall/FrozenLakeModified-v2', 'Frozen Lake (Alternate Reward)', gamma=0.9995, delta=0.0001)

	### DIFFERENT SIZE GRIDS ###

	# register a few different sizes of Frozen Lake...
	registration.register(
		id='ewall/FrozenLakeModified-v10',
		entry_point='env_frozen_lake_mod:FrozenLakeModified',
		kwargs={'map_size': 10, 'map_prob': 0.9, 'is_slippery': True, 'alt_reward': True},
		max_episode_steps=MAX_ITER,
	)
	run_and_evaluate('ewall/FrozenLakeModified-v10', 'Frozen Lake (10x10)',
	                 gamma=0.9995, delta=0.0001,
	                 print_grids=False, plot=False)

	registration.register(
		id='ewall/FrozenLakeModified-v20',
		entry_point='env_frozen_lake_mod:FrozenLakeModified',
		kwargs={'map_size': 20, 'map_prob': 0.9, 'is_slippery': True, 'alt_reward': True},
		max_episode_steps=MAX_ITER,
	)
	run_and_evaluate('ewall/FrozenLakeModified-v20', 'Frozen Lake (20x20)',
	                 gamma=0.9995, delta=0.0001,
	                 print_grids=False, plot=False)

	registration.register(
		id='ewall/FrozenLakeModified-v30',
		entry_point='env_frozen_lake_mod:FrozenLakeModified',
		kwargs={'map_size': 30, 'map_prob': 0.9, 'is_slippery': True, 'alt_reward': True},
		max_episode_steps=MAX_ITER,
	)
	run_and_evaluate('ewall/FrozenLakeModified-v30', 'Frozen Lake (30x30)',
	                 gamma=0.9995, delta=0.0001,
	                 print_grids=False, plot=False)
