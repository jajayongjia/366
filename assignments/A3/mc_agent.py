#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
		   for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""
from utils import rand_in_range, rand_un
import numpy as np
import pickle


e = 0.1
last_state = None
last_action = None
Q = None
returns = None
pi = None
t = None
def agent_init():
	"""
	Hint: Initialize the variables that need to be reset before each run begins
	Returns: nothing
	"""
	global last_state,last_action ,Q ,returns ,pi ,t
	last_state = 0
	last_action = 0	
	returns = np.zeros((99,99)) #Returns(s, a) empty list
	t = np.zeros((99,99))
	
	Q = np.zeros((99,99)) # Q(s, a) arbitrary
	for s in range(1,100):
		for a in range(1,100):
			Q[s-1][a-1] = 1.0
        # Q(s, a) arbitrary
	
	pi = np.zeros((99,99))
	for s in range(1,100):
		a = min(s,100-s)
		pi[s-1][a-1] = 1.0 
	# pi(a|s) an arbitrary "e-soft policy"


def agent_start(state):
	"""
	Hint: Initialize the variavbles that you want to reset before starting a new episode
	Arguments: state: numpy array
	Returns: action: integer
	"""
	global last_state,last_action ,Q ,returns ,pi ,t
	s = state[0]
	action =  np.argmax(pi[s-1])+1 #get the action from policy array;
	last_action = action
	last_state = s
	t[s-1][action-1]+=1
	# pick the first action, don't forget about exploring starts 
	return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
	"""
	Arguments: reward: floting point, state: integer
	Returns: action: floating point
	"""
	global last_state,last_action ,Q ,returns ,pi ,t,e
	s = state[0]
	
	action =  np.argmax(pi[s-1]*Q[s-1])+1 
	#(a) Generate an episode using pi
        
	returns[last_state-1][last_action-1] += (Q[s-1][action-1])
	# G return following the first occurrence of s, a
	# Append G to Returns(s, a)
	
	t[s-1][action-1]+=1
	last_state = s
	last_action = action
	return action

def agent_end(reward):
	"""
	Arguments: reward: floating point
	Returns: Nothing
	"""
	# do learning and update pi
	global last_state,last_action ,Q ,returns ,pi ,t,e
	
	returns[last_state-1][last_action-1] += (reward)
	# do learning 
	
	np.seterr(divide='ignore',invalid='ignore')
	Q = np.nan_to_num(returns/t)
	# updating Q
	# Q(s, a) average(Returns(s, a))
	
       
        #(b) For each pair s, a appearing in the episode:
	for s in range(1,100):	
		
		A = np.argmax(Q[s-1])+1
	        # A = arg maxa Q(s, a)
		
		for a in range(1,min(s,100-s)):
			if a == A:
				pi[s-1][a-1] = 1 - e + e/(min(s,100-s))
			else:
				pi[s-1][a-1] = e/(min(s,100-s))
	        #For all a 2 A(s):
			#pi(a|s) = 1 "+"/|A(s)| "/|A(s)| when a = A
                        #pi(a|s) = e/A(s) when a != A
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






