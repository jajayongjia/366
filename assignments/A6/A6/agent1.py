#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import *
import numpy as np
import pickle

#epsilon = 0.1
alpha =0.5
gamma = 0.9

#Q = None
last_action = None
last_state = None
action = None
num_total_states = 1000 # num_total_states: integer
#model = None
action_number = 2
groupTotalNumber = 10
w = np.zeros(num_total_states)
x = np.identity(num_total_states)
groupSize = 100
#x = None

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    global last_state,w,groupSize,groupTotalNumber,states,x
    last_state = 500
    last_action = 0
    w = np.zeros(num_total_states)
    x = np.identity(num_total_states)
    
def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    global last_state,w,groupSize, last_action
    
    
    if np.random.binomial(1, 0.5) == 1:
        direction = 1
    direction = -1
    
    action = np.random.randint(1, groupSize + 1)
    action *= direction
    

    last_action = action 
    last_state = 500 + action
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global  last_state,w,groupSize,last_action,x
    
    #groupIndex = (state - 1) // groupSize
    #lastGroupIndex = (last_state-1)//groupSize
    w[last_state-1] += alpha*(reward+np.dot(w, x[state-1]) -np.dot(w, x[last_state-1]))
    last_state = state
    
    if np.random.binomial(1, 0.5) == 1:
        direction = 1
    else:
        direction = -1
    
    action = np.random.randint(1, groupSize + 1)
    action *= direction
    

    last_action = action 

    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """ 
    # do learning and update pi
    global last_state,w,groupSize, last_action,x
    
    w[last_state-1] += alpha * (reward + 0 - np.dot(w, x[last_state-1]))
    
   
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return w
    elif (in_message == "requestValueFunction"):
        return w
    else:
        return "I don't know what to return!!"

