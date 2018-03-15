#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

epsilon = 0.1
alpha = 0.5
Q = None
last_action = None
last_state = None
action = None

#action_number = 4
#actions = {0:[1,0],1:[-1,0],2:[0,1],3:[0,-1]}

#uncomment above code if you want to test 4 actions

action_number = 8  # you may change the action_number to 9 
actions = {0:[1,0],1:[1,-1],2:[0,-1],3:[-1,-1],4:[-1,0],5:[-1,1],6:[0,1],7:[1,1]}


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    global Q,last_state,action_number,actions
    last_state = [0,0] # last_state = (x,y)
    last_action = 0
    if action_number == 9:
	actions[8] = [0,0]
    Q = np.zeros((10,7,action_number)) # Q(state[col][row], action)
    
def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    global Q,epsilon,last_action,action,last_state,action_number,actions
    #The state passed through agent_start is always (0,3)
    rand_num = np.random.random() 
       #do e-greedy method
    if epsilon>rand_num:     
	index = rand_in_range(action_number)
    else:
	index = np.argmax(Q[state[0]][state[1]]) #Do greedy
    last_action = index  #last_action is the index 
    action = actions[index]
    last_state = state
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global Q,epsilon,last_action,action,last_state,alpha,action_number,actions
    rand_num = np.random.random() 
       #do e-greedy method
    if epsilon>rand_num:     
	index = rand_in_range(action_number)
    else:
	index = np.argmax(Q[state[0]][state[1]]) #Do greedy
	
    Q[last_state[0]][last_state[1]][last_action] += alpha*(reward + Q[state[0]][state[1]][index] - Q[last_state[0]][last_state[1]][last_action] )
    last_state = state
    last_action = index
    action = actions[index]
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global Q,epsilon,last_action,action,last_state,alpha,actions
    Q[last_state[0]][last_state[1]][last_action] += alpha*(reward - Q[last_state[0]][last_state[1]][last_action] )
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
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

