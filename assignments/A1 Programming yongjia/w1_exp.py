#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

  Experiment runs 2000 runs, each 1000 steps, of an n-armed bandit problem
"""

from rl_glue import *  # Required for RL-Glue
RLGlue("w1_env", "w1_agent")

import numpy as np
import sys

def save_results(data, data_size, filename): # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))
def getOptimalAction():
    return int( RL_env_message("get optimal action") )
def getOption2():
    RL_agent_message("2")
if __name__ == "__main__":
    num_runs = 2000
    max_steps = 1000

    # array to store the results of each step
    optimal_action = np.zeros(max_steps)

    print "\nPrinting one dot for every run: {0} total Runs to complete".format(num_runs)
    for s in range(2):
        
        for k in range(num_runs):
            RL_init()
            if s == 1:
                getOption2()        
            RL_start()
            for i in range(max_steps):
                # RL_step returns (reward, state, action, is_terminal); we need only the
                # action in this problem
                action = RL_step()[2]
    
                '''
                check if action taken was optimal
    
                you need to get the optimal action; see the news/notices
                announcement on eClass for how to implement this
                '''
                bestAction = getOptimalAction()
                if bestAction == int(action):
                    optimal_action[i]+=1
                # update your optimal action statistic here
    
            RL_cleanup()
            print ".",
            sys.stdout.flush()
    
        save_results(optimal_action / num_runs, max_steps, str(s)+"RL_EXP_OUT.dat")
        print "\nDone"