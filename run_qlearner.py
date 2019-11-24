# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import pickle
import gym
import pandas as pd
import env_caveman_world  # registers 'ewall/CavemanWorld-v1' env
import env_frozen_lake_mod  # registers 'ewall/FrozenLakeModified-v1' & v2 (alternate reward) envs
from run_vi_and_pi import evaluate_policy
from qlearner import *

SEED = 1


def run_and_evaluate(env_name, max_iterations=10 ** 7, print_grids=True):

	env = gym.make(env_name)
	env.seed(SEED)
	s = env.reset()
	print('== {} =='.format(env_name))

	# build Q-learner
	ee = greedy()
	# ee = eps_greedy(0.8)
	# ee = eps_decay(decay=0.00005, verbose=True)
	# ee = greedy_decay(verbose=False)
	ql = QLearner(num_states=env.observation_space.n,
	              num_actions=env.action_space.n,
	              random_explorer=ee,
	              alpha=0.9,
	              gamma=0.9995,
	              optimistic_init=0.01,
	              verbose=False)
	ql.reset(s)

	# run learner
	policy, iters, q_variation, episode_rewards = ql.run(env, max_iterations=max_iterations)

	print('\n== Q-Learning ==')
	print('Iterations:', iters)
	# print('Variations:', q_variation)
	# print('Rewards curve:', episode_rewards)
	print()

	print('== QL Policy ==')
	if print_grids:
		env.print_policy(policy)
	ql_scores, ql_steps = evaluate_policy(env, policy)
	print('Average total reward:', np.mean(ql_scores), 'max reward:', np.max(ql_scores))
	print('Average steps:', np.mean(ql_steps), 'max steps:', np.max(ql_steps), '\n')

	return policy


def run_ee_comparison(env_name, max_iterations=10 ** 7, print_grids=True):
	"""" Compare different exploit/explore methods """
	print('== {}: Explore/Exploit Experiments ==\n'.format(env_name))

	env = gym.make(env_name)
	env.seed(SEED)

	# prepare to save results
	result_labels = ('ee_method', 'percent_random', 'mean_reward', 'max_reward', 'mean_actions', 'max_actions')
	results = []

	# prepare EE methods
	ee_methods = {'optimistic': greedy(),
	              'eps_greedy': eps_greedy(0.8),
	              'eps_decay': eps_decay(decay=0.000005)}
	# skipping: 'greedy_decay': greedy_decay()

	# loop and evaluate each method
	for name, ee in ee_methods.items():
		print('-- Method:', name)

		# optimism-in-the-face-of-uncertainty must initialize Q > 0
		opt_init = 0.01 if name == 'optimistic' else None

		# build a Q-learner
		ql = QLearner(num_states=env.observation_space.n,
		              num_actions=env.action_space.n,
		              random_explorer=ee,
		              alpha=0.9,
		              gamma=0.9995,
		              optimistic_init=opt_init,
		              verbose=False)

		# reset environtment and run learner
		s = env.reset()
		ql.reset(s)
		policy, _, _, _ = ql.run(env, max_iterations=max_iterations)

		print('Policy:')
		if print_grids:
			env.print_policy(policy)
		print()

		# evaluate
		percent_random = ee.get_percent_randomized()
		print('Percent randomly explored:', percent_random)

		ql_scores, ql_actions = evaluate_policy(env, policy, n=100)
		mean_reward, max_reward = np.mean(ql_scores), np.max(ql_scores)
		print('Average total reward:', mean_reward, '| max reward:', max_reward)
		mean_actions, max_actions = np.mean(ql_actions), np.max(ql_actions)
		print('Average actions taken:', mean_actions, '| max actions:', max_actions, '\n')

		results.append((name, percent_random, mean_reward, max_reward, mean_actions, max_actions))

	results = pd.DataFrame(results, columns=result_labels)
	return results


if __name__ == "__main__":
	# seed pseudo-RNG for reproducibility
	random.seed(SEED)
	np.random.seed(SEED)

	# allow Pandas to print more
	pd.options.display.width = 0

	# # run Caveman's World (simple problem)
	# run_and_evaluate('ewall/CavemanWorld-v1', 100)

	# # run Frozen Lake (Original)
	# run_and_evaluate('ewall/FrozenLakeModified-v1', 5 * 10 ** 8)
	#
	# # run Frozen Lake (Alternate)
	# run_and_evaluate('ewall/FrozenLakeModified-v2', 5 * 10 ** 8)

	# run explore/exploit comparison on Frozen Lake/Original
	ee_fl_orig = run_ee_comparison('ewall/FrozenLakeModified-v1', max_iterations=10 ** 7)
	print(ee_fl_orig)
	pickle.dump(ee_fl_orig, open('pickles/ee_fl_orig', 'wb'))

	# run explore/exploit comparison on Frozen Lake/Modified
	ee_fl_alt = run_ee_comparison('ewall/FrozenLakeModified-v2', max_iterations=10 ** 7)
	print(ee_fl_alt)
	pickle.dump(ee_fl_alt, open('pickles/ee_fl_alt', 'wb'))
