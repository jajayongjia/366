#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

head_probability = 0.55 # head_probability: floating point
num_total_states = 99 # num_total_states: integer
current_state = None
wind_affect = 0
current_state = None

def env_init():
    global current_state
    current_state = np.zeros(2) # current state is in matrix format where current_state[col][row]


def env_start():
    """ returns numpy array """
    global current_state
    current_state = [0,3] # the start position
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state,wind_affect
    
    #First Check valid action:
    a_x = action[0]
    a_y = action[1]
    s_x = current_state[0]
    s_y = current_state[1]
    new_x = a_x + s_x
    new_y = a_y + s_y
    
    if new_x in (3,4,5,8):
        new_y+=1
    elif new_x in (6,7):
        new_y+=2
    elif new_x > 9:
        new_x = 9
    elif new_x <0:
        new_x = 0    
    if new_y >6:
        new_y = 6
    elif new_y <0:
        new_y = 0
    current_state = [new_x,new_y]
    if current_state == [7,3]:
        reward = +1
        is_terminal = True
    else:
        reward = -1
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
