#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from utils import rand_in_range
import numpy as np

last_action = None # last_action: NumPy array

num_actions = 10

def agent_init():
    global last_action
    global option
    option = "1"
    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action
    global option
    global Q
    global a 
    a = 0.1
    global e
    if option == "2":
        Q = np.zeros(10) 
        for n in range(10):
            Q[n] = 5.0
        e = 0.0
    else:
        Q = np.zeros(10) 
        e = 0.1
    
    
    last_action[0] = rand_in_range(num_actions)

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)

    return local_action[0]


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action
    global Q
    global e
    
    Q[int(last_action)] += 0.1*(reward - Q[int(last_action)])
    #do the learning
    local_action = np.zeros(1)
    rand_num = np.random.random() 
    #do e-greedy method
    if e>rand_num: 
        local_action[0] = rand_in_range(num_actions)
    else: 
        local_action[0] = np.argmax(Q)


    last_action = local_action

    return last_action

def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    return

def agent_cleanup():
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent
    global option
    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
    if inMessage == "2":
        option = "2"
    else:
        return "I don't know how to respond to your message"