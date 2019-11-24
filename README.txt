Project 4: Reinforcement Learning
#################################
GT CS7641 Machine Learning, Fall 2019
Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196


## Background ##
Classwork for Georgia Tech's CS7641 Machine Learning course. Project code should be published publicly for grading
purposes, under the assumption that students should not plagiarize content and must do their own analysis.

These project contains implementations of 3 reinforcement learning algorithms (value iteration, policy iteration,
and Q-learning) and 2 MDP problems (Caveman's World and large customized Frozen Lake). The experiments here are
meant to help compare and contrast these algorithms for a written analysis paper (which is *not* included in this
repository, for obvious reasons!).


## Requirements ##

* Python 3.7 or higher
* Python libraries such as Numpy, Pandas, Matplotlib, and OpenAI Gym


## Instructions ##

* Clone this source repository from Github onto your computer using the following command:
	`git clone git@github.com:ewall/CS7641_Reinforcement_Learning.git`

* From the source directory, run the following command to install the necessary Python modules:
	`pip install -r requirements.txt`

* To execute the experiments for value iteration and policy iteration, run:
    `python run_vi_and_pi.py`

* To execute the experiments for Q-learning, run the following:
    `python run_qlearning.py`

* Images for plots will be created in the "plots" directory.

* The 'env_caveman_world.py' and 'env_frozen_lake_mod.py' files are imported by the run scripts.
