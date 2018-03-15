#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

#head_probability = 0.55 # head_probability: floating point
num_total_states = 1000 # num_total_states: integer
current_state = None
goal_state =  [0, num_total_states + 1]
trueStateValues = np.arange(-1001, 1003, 2) / 1001.0
ACTIONS = [-1,1]
STEP_RANGE = 100

# Dynamic programming to find the true state values, based on the promising guess above
# Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states

def env_init():
    global current_state
    current_state = 500
  

def env_start():
    """ returns numpy array """
    global current_state
    current_state = 500
    return current_state

def env_step(action):
    global current_state,goal_state
    
    current_state +=action
    
    if current_state >=goal_state[1]:
        reward = 1
        is_terminal = True
    elif current_state <= goal_state[0]:
        is_terminal = True
        reward = -1
    else:
        reward = 0
        is_terminal = False
    
    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
