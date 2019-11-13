# Project 4: Reinforcement Learning -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

# pip install git+https://github.com/BlackHC/mdp.git
from blackhc import mdp
from blackhc.mdp import dsl


def caveman_world():
	with dsl.new() as mdp:
		hungry = dsl.state('hungry')
		got_food = dsl.state('got food')
		full = dsl.state('full')
		dead = dsl.terminal_state('dead')

		sleep = dsl.action('sleep')
		hunt = dsl.action('hunt')
		eat = dsl.action('eat')

		hungry & sleep > hungry * 7 | dead * 3
		hungry & hunt > got_food * 9 | dead
		hungry & eat > hungry

		got_food & sleep > hungry * 2 | got_food * 8
		got_food & eat > hungry *2 | full * 8
		got_food & hunt > got_food

		full & sleep > hungry
		full & eat > dead
		# full & eat > dsl.reward(-10)
		full & hunt > full

		return mdp.validate()

if __name__ == "__main__":
	CAVEMAN_WORLD = caveman_world()
	env = CAVEMAN_WORLD.to_env()
	env.reset()
	env.render()
