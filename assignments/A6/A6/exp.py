
####!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""
from utils import *
from rl_glue import *  # Required for RL-Glue
import sys
import time
import math
import numpy as np
import pickle
from rndmwalk_policy_evaluation import compute_value_function
import matplotlib.pyplot as plt

agents = ["agent1","agent2","agent3"]
for agent in agents:
	RLGlue("env", agent)	
	if __name__ == "__main__":
		total_episodes =2000
		print("Doing " + agent)
		V = np.load('TrueValueFunction.npy')
		runs = 10
		errors = np.zeros(total_episodes//10)
		np.random.seed()
		for run in range(0, runs):
			print("Run : " +str(run+1))
			RL_init()
			for num_episodes in range (total_episodes) :	
				RL_episode(20000)  
				if (num_episodes//10 - (num_episodes-1)//10) != 0:
					stateValues = RL_agent_message("ValueFunction")
					errors[num_episodes//10] += math.sqrt(sum(np.power(V[1:] - stateValues, 2))/1000)
				RL_cleanup()
		errors = errors/runs
		plt.figure(agents.index(agent)+1)
		plt.plot(errors)
		plt.xlabel('Episodes')
		plt.ylabel('RMSVE')
		plt.show()		
		if agent == "agent3":
			print("Done ! ")
