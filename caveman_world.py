# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import sys
from contextlib import closing

import numpy as np
from gym.envs.toy_text import discrete


class CavemanWorldEnv(discrete.DiscreteEnv):
	"""
	ewall/CavemanWorld-v1

	...Description TBD...

	states:  {0: "hungry", 1: "got food", 2: "full", 3: "dead"}
	actions: {0: "sleep", 1: "hunt", 2: "eat"}
	rewards = {"hungry": 0, "got food": 1, "full": 10, "dead": -10}

	"""

	metadata = {'render.modes': ['human']}

	def __init__(self):

		nA = 3
		nS = 4
		P = {0: {0: [(0.7, 0, 0, False), (0.3, 3, -10, True)], 1: [(0.1, 3, -10, True), (0.9, 2, 1, False)], 2: [(1.0, 0, 0, False)]},
		     1: {0: [(0.2, 0, 0, False), (0.8, 1, 1, False)], 1: [(1.0, 1, 0, False)], 2: [(0.2, 0, 0, False), (0.8, 2, 10, False)]},
		     2: {0: [(1.0, 0, 0, False)], 1: [(1.0, 2, 0, False)], 2: [(1.0, 3, -10, True)]},
		     3: {0: [(1.0, 3, 0, True)], 1: [(1.0, 3, 0, True)], 2: [(1.0, 3, 0, True)]}}
		#TODO should "dead" be a state, or just an ending?
		#TODO how to handle invalid actions (e.g. you shouldn't be able to "hunt" from "full")
		isd = np.zeros(nS)

		self.actions_text = {0: "sleep", 1: "hunt", 2: "eat"}
		self.states_text = {0: "hungry", 1: "got food", 2: "full", 3: "dead"}
		self.reward_range = (-10, 10)

		super(CavemanWorldEnv, self).__init__(nS, nA, P, isd)

	def render(self, mode='human'):
		outfile = sys.stdout

		if self.lastaction is not None:
			outfile.write("  ({})\n".format(["Sleep", "Hunt", "Eat"][self.lastaction]))
		else:
			outfile.write("\n")

		# TODO print some useful output here

		with closing(outfile):
			return outfile.getvalue()

	def print_grid(self):
		""" N/A for this problem type"""
		print()

	def print_policy(self, policy):
		""" Pretty print given policy """
		for s, a in zip(self.states_text.values(), [self.actions_text[action] for action in policy]):
			print("State: %s --> Action: %s" %(s, a))
		print()
