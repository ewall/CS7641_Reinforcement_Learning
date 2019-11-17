# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import sys

import numpy as np
from gym.envs import registration
from gym.envs.toy_text import discrete

MAX_ITER = 10 ** 6


class CavemanWorldEnv(discrete.DiscreteEnv):
	"""
	ewall/CavemanWorld-v1

	You're a caveman, and your world is simple but dangerous. Each day you can choose to eat, sleep, or hunt... but
	your decisions have consequences. For example, sleeping when you're hungry or eating when you're already full
	could lead to death.

	This is an easy, non-deterministic environment with a known optimal policy.

	The problem is described in Jonathan Scholz's "Markov Decision Processesand Reinforcement Learning: An Introduction
	to Stochastic Planning" which (at the time of this writing) is available at this URL:
	https://s3.amazonaws.com/ml-class/notes/MDPIntro.pdf

	Modified the original by creating new transitions for states omitted in the diagram:
	• when hungry, eating provides -10 reward and leaves you still hungry
	• when got food, hunting has a 90% chance of getting more food with no reward, and a 10% chance of killing you
	• when full, hunting will 100% kill you

	These modifications do not change the optimal policy, which remains as follows:
	• state: hungry --> action: hunt
	• state: got food --> action: eat
	• state: full --> action: sleep
	• state: dead --> action: sleep
	"""

	metadata = {'render.modes': ['human']}

	def __init__(self):

		nA = 3
		nS = 4

		# dictionaries to make P easier to read
		S = {'hungry': 0, 'got food': 1, 'full': 2, 'dead': 3}
		A = {'sleep': 0, 'hunt': 1, 'eat': 2}
		R = {'hungry': 0, 'got food': 1, 'full': 10, 'dead': -10}
		D = {'hungry': False, 'got food': False, 'full': False, 'dead': True}

		# for reference: P[s][a] = [(prob, s',r, done), (prob, s', r, done)...]
		P = {S['hungry']: {A['sleep']: [(0.7, S['hungry'], R['hungry'], False), (0.3, S['dead'], R['dead'], True)],
		                   A['hunt']: [(0.9, S['got food'], R['got food'], False), (0.1, S['dead'], R['dead'], True)],
		                   A['eat']: [(1.0, S['hungry'], -10, False)]},
		     S['got food']: {A['sleep']: [(0.8, S['got food'], 0, False), (0.2, S['hungry'], R['hungry'], False)],
		                     A['hunt']: [(0.9, S['got food'], 0, False), (0.1, S['dead'], R['dead'], True)],
		                     A['eat']: [(0.8, S['full'], R['full'], False), (0.2, S['hungry'], R['hungry'], False)]},
		     S['full']: {A['sleep']: [(1.0, S['hungry'], R['hungry'], False)],
		                 A['hunt']: [(1.0, S['dead'], R['dead'], True)],
		                 A['eat']: [(1.0, S['dead'], R['dead'], True)]},
		     S['dead']: {A['sleep']: [(1.0, S['dead'], 0, True)],
		                 A['hunt']: [(1.0, S['dead'], 0, True)],
		                 A['eat']: [(1.0, S['dead'], 0, True)]}}

		isd = np.array((1.0, 0.0, 0.0, 0.0))  # initial state description: what states can we start from?

		self.actions_text = {0: "sleep", 1: "hunt", 2: "eat"}
		self.states_text = {0: "hungry", 1: "got food", 2: "full", 3: "dead"}

		self.reward_range = (-10, 10)

		self.optimal_policy = np.array((1, 2, 0, 0))

		super(CavemanWorldEnv, self).__init__(nS, nA, P, isd)

	def print_grid(self):
		""" n/a for this problem type """
		pass

	def print_policy(self, policy):
		""" Pretty print given policy """
		for s, a in zip(self.states_text.values(), [self.actions_text[action] for action in policy]):
			print("@ state: %s --> action: %s" % (s, a))

	def render(self, mode='human'):
		outfile = sys.stdout

		if self.lastaction is not None:
			outfile.write("state: %s --> action: %s\n" %(self.actions_text[self.lastaction], self.states_text[self.s]))
		else:
			outfile.write("\n")


### the following should run when module is imported ###

# register this gym env
registration.register(
	id='ewall/CavemanWorld-v1',
	entry_point='caveman_world:CavemanWorldEnv',
	max_episode_steps = MAX_ITER,
)
