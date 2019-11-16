# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import sys
from contextlib import closing

import numpy as np
from six import StringIO  # , b

from gym import utils
from gym.envs import registration
from gym.envs.toy_text import discrete
from gym.envs.toy_text.frozen_lake import generate_random_map, LEFT, DOWN, RIGHT, UP

FROZEN_PROB = 0.9
GRID_SIZE = 30
MAX_ITER = 10 ** 6
SLIPPERY = True


### Code Credit: the following is only slightly modified from the original source:
#   Title: OpenAI Gym - FrozenLake-v0 environment
#   Author: Greg Brockman; Vicki Cheung; Ludwig Pettersson; Jonas Schneider; John Schulman; Jie Tang; Wojciech Zaremba
#   Date: 2019-11-08
#   Code version: v0.15.4
#   Availability: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py


class FrozenLakeModified(discrete.DiscreteEnv):
	"""
	ewall/FrozenLakeModified-v1

	Winter is here. You and your friends were tossing around a frisbee at the park
	when you made a wild throw that left the frisbee out in the middle of the lake.
	The water is mostly frozen, but there are a few holes where the ice has melted.
	If you step into one of those holes, you'll fall into the freezing water.
	At this time, there's an international frisbee shortage, so it's absolutely imperative that
	you navigate across the lake and retrieve the disc.
	However, the ice is slippery, so you won't always move in the direction you intend.
	The surface is described using a grid like the following

		SFFF
		FHFH
		FFFH
		HFFG

	S : starting point, safe
	F : frozen surface, safe
	H : hole, fall to your doom
	G : goal, where the frisbee is located

	The episode ends when you reach the goal or fall in a hole.

	* REWARDS HAVE BEEN CHANGED *

	You receive a reward of 100 if you reach the goal, and -1 otherwise. (This incentivizes taking the shortest path.)
	"""

	metadata = {'render.modes': ['human', 'ansi']}

	def __init__(self, map_size=30, map_prob=0.9, is_slippery=True, alt_reward=True):
		desc = generate_random_map(size=map_size, p=map_prob)
		self.desc = desc = np.asarray(desc, dtype='c')
		self.nrow, self.ncol = nrow, ncol = desc.shape
		self.mapping = {0: "◄", 1: "▼", 2: "►", 3: "▲"}  # {0: "←", 1: "↓", 2: "→", 3: "↑"}

		if alt_reward:
			self.reward_range = (-1, 100)
		else:
			self.reward_range = (0, 1)

		nA = 4
		nS = nrow * ncol

		isd = np.array(desc == b'S').astype('float64').ravel()
		isd /= isd.sum()

		P = {s: {a: [] for a in range(nA)} for s in range(nS)}

		def to_s(row, col):
			return row * ncol + col

		def inc(row, col, a):
			if a == LEFT:
				col = max(col - 1, 0)
			elif a == DOWN:
				row = min(row + 1, nrow - 1)
			elif a == RIGHT:
				col = min(col + 1, ncol - 1)
			elif a == UP:
				row = max(row - 1, 0)
			return row, col

		for row in range(nrow):
			for col in range(ncol):
				s = to_s(row, col)
				for a in range(4):
					li = P[s][a]
					letter = desc[row, col]
					if letter in b'GH':
						li.append((1.0, s, 0, True))
					else:
						if is_slippery:
							for b in [(a - 1) % 4, a, (a + 1) % 4]:
								newrow, newcol = inc(row, col, b)
								newstate = to_s(newrow, newcol)
								newletter = desc[newrow, newcol]
								done = bytes(newletter) in b'GH'
								if alt_reward:
									if newletter == b'G':
										rew = 100.0
									else:
										rew = -1.0
								else:
									rew = float(newletter == b'G')
								li.append((1.0 / 3.0, newstate, rew, done))
						else:
							newrow, newcol = inc(row, col, a)
							newstate = to_s(newrow, newcol)
							newletter = desc[newrow, newcol]
							done = bytes(newletter) in b'GH'
							if alt_reward:
								if newletter == b'G':
									rew = 100.0
								else:
									rew = -1.0
							else:
								rew = float(newletter == b'G')
							li.append((1.0, newstate, rew, done))

		super(FrozenLakeModified, self).__init__(nS, nA, P, isd)

	def render(self, mode='human'):
		outfile = StringIO() if mode == 'ansi' else sys.stdout

		row, col = self.s // self.ncol, self.s % self.ncol
		desc = self.desc.tolist()
		desc = [[c.decode('utf-8') for c in line] for line in desc]
		desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
		if self.lastaction is not None:
			outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
		else:
			outfile.write("\n")
		outfile.write("\n".join(''.join(line) for line in desc) + "\n")

		if mode != 'human':
			with closing(outfile):
				return outfile.getvalue()

	def print_grid(self):
		""" Pretty print the current grid"""
		print('Grid:')
		print('\n'.join([''.join([str(cell, "utf-8") for cell in row]) for row in self.desc]))
		print()

	def print_policy(self, policy):
		""" Pretty print a given policy """
		pol = np.array([self.mapping[action] for action in policy]).reshape(self.desc.shape).tolist()
		# pol[0][0] = 'S'
		pol[-1][-1] = 'G'

		for row in range(len(pol)):
			for col in range(len(pol[0])):
				if self.desc[row][col] == b'H':
					pol[row][col] = 'O'

		print('\n'.join([''.join([str(cell) for cell in row]) for row in pol]))
		print()


# register this gym env when module is imported
registration.register(
	id='ewall/FrozenLakeModified-v1',
	entry_point='frozen_lake_mod:FrozenLakeModified',
	kwargs={'map_size': GRID_SIZE, 'map_prob': FROZEN_PROB, 'is_slippery': SLIPPERY},
	max_episode_steps=MAX_ITER,
)
