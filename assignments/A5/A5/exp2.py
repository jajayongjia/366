#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""
from utils import *
from rl_glue import *  # Required for RL-Glue
RLGlue("env", "agent")
import sys
import time
import numpy as np
import pickle

start_time = time.time()
if __name__ == "__main__":
	total_episodes = 50
	max_runs = 10
	aList = [0.03125,0.0625,0.125,0.25,0.5,1.0]
	nList =  [5]
	np.random.seed(100)
	data_file = open("result2.txt", "w")
	for a in aList:
		print("a is " + str(a))
		set_a(a)
		data = np.zeros(50)
		for n in nList:	
			set_n(n)
			for current_runs in range(max_runs) :
				print("run number : " + str(current_runs))
				RL_init()
				for num_episodes in range (total_episodes) :
					RL_episode(1800)
					steps = RL_num_steps()
					data[num_episodes] += steps
				RL_cleanup()	
			results = data/500
			data_file.write(str( sum(results))+",")
			data_file.write("--")
	data_file.close()
	print ("Done")

