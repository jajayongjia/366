#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("windyGridworld_env", "windyGridworld_agent")
import sys
import time
import numpy as np
import pickle


start_time = time.time()
if __name__ == "__main__":
	num_episodes = 0
	max_steps = 8000
	current_steps = 0
	data_file = open("result.txt", "w")
	RL_init()
	while current_steps<8000:
		RL_episode(10000)
		current_steps += RL_num_steps()
		num_episodes = RL_num_episodes()
		data_file.write(str(current_steps)+','+str(num_episodes)+"\n")
	RL_cleanup()
	data_file.close()
	print("Done")