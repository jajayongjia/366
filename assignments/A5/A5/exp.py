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
	nList =  [0,5,50]
	a = 0.1
	set_a(a)
	np.random.seed(100)
	data_file = open("result.txt", "w")
	for n in nList:	
		set_n(n)
		print("n is : " + str(n))
		data = np.zeros(50)
		for current_runs in range(max_runs) :
		        print("run number : " + str(current_runs))
			RL_init()
			for num_episodes in range (total_episodes) :
				RL_episode(1800)
				steps = RL_num_steps()
				data[num_episodes] += steps
			RL_cleanup()
		data = data/10
		for i in data:
			data_file.write(str(i)+",")
                data_file.write("--")
	data_file.close()
	print ("Done")

