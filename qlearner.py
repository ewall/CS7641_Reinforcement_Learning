# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import random
from math import log, sqrt
import numpy as np
from run_vi_and_pi import diff_policies, timing

MAX_ITER = 10 ** 6


class QLearner(object):

	### Code Credit: this code is my own implementation from Fall 2018, but portions are built off this template:
	#   Title: Template for implementing QLearner
	#   Author: Tucker Balch
	#   Date: 2018
	#   Availability: only via Georgia Tech course CS7646

	def __init__(self,
	             num_states,
	             num_actions,
	             random_explorer=None,
	             alpha=0.2,
	             gamma=0.9,
	             optimistic_init=None,
	             verbose=False):
		"""
		Initialize QLearner object.
		:param num_states: Total number of possible states; integer.
		:param num_actions: Total number of possible actions; integer.
		:param random_explorer: Object for randomized exploration/exploitation choices.
		:param alpha: Learning rate; float between 0.0 and 1.0, or None to use built-in decay.
		:param gamma: Discount rate: float between 0.0 and 1.0
		:param optimistic_init: Initialize Q for optimism-in-the-face-of-uncertainty; float.
		:param verbose: Enable debugging printouts.
		"""

		# sanity checks
		if gamma < 0 or gamma > 1:
			raise ValueError("gamma value must be between 0.0 and 1.0.")
		if alpha is not None:
			if alpha < 0 or alpha > 1:
				raise ValueError("alpha value must be between 0.0 and 1.0.")

		# save config
		self.num_actions = num_actions
		if random_explorer is not None:
			self.random_explorer = random_explorer
		else:
			self.random_explorer = eps_decay()
		self.alpha = alpha
		self.gamma = gamma
		self.verbose = verbose

		# set the stage
		self.s = None  # previous state
		self.a = None  # previous action

		# the all-important Q table
		if optimistic_init is not None:
			self.q = np.full((num_states, num_actions), float(optimistic_init))
		else:
			self.q = np.zeros((num_states, num_actions))

	def get_policy(self):
		""" Return current best policy """
		return self.q.argmax(axis=1)

	def get_v(self):
		""" Return current best V """
		return self.q.max(axis=1)

	def reset(self, s=0):
		""" Reset to initial state and clear counters"""
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
		""" Iterate thru episodes with gym env until stopped """
		optimal_achieved = False
		initial_state = env.reset()
		action = self.reset(initial_state)  # set the initial state and get first action

		for i in range(max_iterations):
			state, reward, done, _ = env.step(action)
			action = self.update_and_query(state, reward, i)

			if done:
				if self.verbose:
					print("   episode done")
				initial_state = env.reset()
				self.reset(initial_state)
				continue

			# check if optimal policy already achieved
			if hasattr(env, 'optimal_policy') and optimal_achieved == False:
				current_policy = self.get_policy()
				diff = diff_policies(current_policy, env.optimal_policy)
				if diff == 0:
					optimal_achieved = True
					print("Optimal policy found on iteration", str(i + 1))
					# break

			#TODO convergence detection (other than optimal policy) not yet implemented

		if self.verbose:
			print("Q:\n", self.q)

		return self.get_policy(), i + 1

	def update_and_query(self, s_prime, r, n=0):
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
		# previously: future_q = r + self.gamma * self.q[s_prime, np.argmax(self.q[s_prime, :])]
		future_q = r + self.gamma * self.q[s_prime, :].max() - prev_q

		if self.alpha is None:
			# use built-in decay method
			future_q = (1 / sqrt(n + 2)) * future_q
			self.q[self.s, self.a] = prev_q + future_q
		else:
			# use static alpha value
			self.q[self.s, self.a] = (1 - self.alpha) * prev_q + (self.alpha * future_q)

		if self.verbose:
			print("   prev_q:", prev_q, "future_q:", future_q, "q_value:", self.q[self.s, self.a])

		# decide if action will be random
		if isinstance(self.random_explorer, min_explorer):
			self.random_explorer.set_state(self.s, self.a)
		if self.random_explorer.eval():
			action = random.randint(0, self.num_actions - 1)  # random exploration
			if self.verbose:
				print("   a =", action, "(random)")
		else:
			action = np.argmax(self.q[s_prime, :])  # exploit Q table
			if self.verbose:
				print("   a =", action, "(exploiting)")

		# store S' and A' for next time
		self.s = s_prime
		self.a = action

		return action


class greedy(object):
	""" Greedy exploitation only """
	def __init__(self):
		pass

	def eval(self):
		return False

	def get_percent_randomized(self):
		return 0.0


class eps_greedy(object):
	""" Explore epsilon percent of the time """
	def __init__(self, epsilon, verbose=False):
		if epsilon < 0 or epsilon > 1:
			raise ValueError("epsilon value must be between 0.0 and 1.0.")
		self.epsilon = epsilon
		self.verbose = verbose
		self.t = 0
		self.r = 0

	def eval(self):
		pick_randomly = True if random.random() <= self.epsilon else False

		if pick_randomly:
			self.r += 1
		self.t += 1

		if self.verbose:
			print("   random?=", pick_randomly)
		return pick_randomly

	def get_percent_randomized(self):
		return self.r / self.t


class eps_decay(object):
	""" Epsilon-greedy with exponential decay """
	def __init__(self, epsilon=1.0, decay=0.005, minimum=0.0, verbose=False):
		if epsilon < 0 or epsilon > 1:
			raise ValueError("epsilon value must be between 0.0 and 1.0.")
		self.epsilon = self.start = epsilon
		self.decay = decay
		self.minimum = minimum
		self.verbose = verbose
		self.t = 0
		self.r = 0

	def eval(self):
		pick_randomly = True if random.random() <= self.epsilon else False

		if pick_randomly:
			self.r += 1
		self.t += 1

		if self.verbose:
			print("   random?=", pick_randomly, "eps=", self.epsilon, "at t=", self.t)

		# decay the random action rate for next time
		self.epsilon = 1.0 * np.exp(-self.start * self.decay * self.t)
		if self.epsilon < self.minimum:
			self.epsilon = self.minimum

		return pick_randomly

	def get_percent_randomized(self):
		return self.r / self.t


class greedy_decay(object):
	""" Greedy exploitation with logarithmic decay based on iterations """
	def __init__(self, verbose=False):
		self.verbose = verbose
		self.t = 0
		self.r = 0

	def eval(self):
		pick_randomly = np.random.random() >= (1 - (1 / log(self.t + 2)))

		if pick_randomly:
			self.r += 1
		self.t += 1

		if self.verbose:
			print("   random?=", pick_randomly, "at t=", self.t)

		return pick_randomly

	def get_percent_randomized(self):
		return self.r / self.t


class min_explorer(object):
	""" Ensure each state/action pair is explored a minimum number of times """
	def __init__(self, num_states, num_actions, min_explores, verbose=False):
		self.explorations = np.full((num_states, num_actions), int(min_explores))
		self.verbose = verbose
		self.t = 0
		self.r = 0

	def set_state(self, s, a):
		self.s = s
		self.a = a

	def eval(self, s=None, a=None):
		if s is None:
			s = self.s
		if a is None:
			a = self.a

		if self.explorations[s][a] > 0:
			self.explorations[s][a] -= 1
			pick_randomly = True
			self.r += 1
		else:
			pick_randomly = False

		self.t += 1

		if self.verbose:
			print("   random?=", pick_randomly, "at (s,a):", s, a)

		return pick_randomly

	def get_percent_randomized(self):
		return self.r / self.t

	def still_exploring(self):
		return np.any(self.explorations)


if __name__ == "__main__":
	pass
