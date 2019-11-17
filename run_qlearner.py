# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import gym
import env_caveman_world  # registers 'ewall/CavemanWorld-v1' env
import env_frozen_lake_mod  # registers 'ewall/FrozenLakeModified-v1' & v2 (alternate reward) envs
from run_vi_and_pi import evaluate_policy
from qlearner import *

MAX_ITER = 10 ** 6
SEED = 1


def run_and_evaluate(env_name, print_grids=True):

	#TODO add max_iterations to vary by problem?

	env = gym.make(env_name)
	env.seed(SEED)
	s = env.reset()
	print('== {} =='.format(env_name))

	# build Q-learner
	ee = greedy()
	# ee = eps_greedy(0.8)
	# ee = eps_decay(decay=0.00005, verbose=True)
	ql = QLearner(num_states=env.observation_space.n,
	              num_actions=env.action_space.n,
	              random_explorer=ee,
	              alpha=0.9,
	              gamma=0.9,
	              optimistic_init=0.1,
	              verbose=False)
	ql.reset(s)

	# run learner
	policy, iters, q_variation, episode_rewards = ql.run(env, max_iterations=MAX_ITER)

	print('\n== Q-Learning ==')
	print('Iterations:', iters)
	print('Variations:', q_variation)
	print('Rewards curve:', episode_rewards, '\n')

	print('== QL Policy ==')
	if print_grids:
		env.print_policy(policy)
	ql_scores, ql_steps = evaluate_policy(env, policy)
	print('Average total reward:', np.mean(ql_scores), 'max reward:', np.max(ql_scores))
	print('Average steps:', np.mean(ql_steps), 'max steps:', np.max(ql_steps), '\n')

	return policy


	#TODO loop thru different settings for plotting


if __name__ == "__main__":
	# seed pseudo-RNG for reproducibility
	random.seed(SEED)
	np.random.seed(SEED)

	# # run Caveman's World (simple problem)
	# run_and_evaluate('ewall/CavemanWorld-v1')

	# run Frozen Lake Modified with Alternate Rewards  (large grid problem)
	run_and_evaluate('ewall/FrozenLakeModified-v2')
