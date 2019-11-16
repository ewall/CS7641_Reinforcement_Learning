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
	• when hungry, eating provides -1 reward and leaves you hungry
	• when got good, hunting has 10% chance of killing you and 90% chance of getting more food with no reward
	• when full, hunting has 80% chance of killing you and 20% chance of getting more food with no reward

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

		"""
		reference for P data structure: 
			P[s][a] = [(prob, s',r, done), (prob, s', r, done)...]
			states: {0: "hungry", 1: "got food", 2: "full", 3: "dead"}
			rewards = {"hungry": 0, "got food": 1, "full": 10, "dead": -10 & end episode}
			actions: {0: "sleep", 1: "hunt", 2: "eat"}
			#TODO these would be easier to read as enums, of course...
		"""
		P = {0: {0: [(0.7, 0, 0, False), (0.3, 3, -10, True)],
		         1: [(0.1, 3, -10, True), (0.9, 2, 1, False)],
		         2: [(1.0, 0, -1, False)]},
		     1: {0: [(0.2, 0, 0, False), (0.8, 1, 1, False)],
		         1: [(0.1, 3, -10, True), (0.9, 2, 0, False)],
		         2: [(0.2, 0, 0, False), (0.8, 2, 10, False)]},
		     2: {0: [(1.0, 0, 0, False)],
		         1: [(0.8, 3, -10, True), (0.2, 2, 0, False)],
		         2: [(1.0, 3, -10, True)]},
		     3: {0: [(1.0, 3, 0, True)],
		         1: [(1.0, 3, 0, True)],
		         2: [(1.0, 3, 0, True)]}}

		isd = np.zeros(nS)  # initial state description doesn't matter for this problem, but is required by super()

		self.actions_text = {0: "sleep", 1: "hunt", 2: "eat"}
		self.states_text = {0: "hungry", 1: "got food", 2: "full", 3: "dead"}
		self.reward_range = (-10, 10)

		super(CavemanWorldEnv, self).__init__(nS, nA, P, isd)

	def render(self, mode='human'):
		outfile = sys.stdout

		if self.lastaction is not None:
			outfile.write("state: %s --> action: %s\n" %(self.actions_text[self.lastaction], self.states_text[self.s]))
		else:
			outfile.write("\n")

	def print_grid(self):
		""" n/a for this problem type, but does print a newline """
		print()

	def print_policy(self, policy):
		""" Pretty print given policy """
		for s, a in zip(self.states_text.values(), [self.actions_text[action] for action in policy]):
			print("state: %s --> action: %s" % (s, a))
		print()

	def reset(self):
		self.s = 0
		self.lastaction = None
		return self.s


# register this gym env when module is imported
registration.register(
	id='ewall/CavemanWorld-v1',
	entry_point='caveman_world:CavemanWorldEnv',
	max_episode_steps = MAX_ITER,
)
