import numpy as np
from importlib import import_module
from tiles3 import *
import random

# the format of  state returned from env program is np,array[position,velocity]
# the init position is a random number 
# and the init velocity is 0

min_position = -1.2
max_position = 0.5
min_velocity = -0.07
max_velocity = 0.07
actions = [0,1,2]

memorySize = 4096
num_tilings = 8
tili_size = 8
alpha = 0.1/float(num_tilings)
lada = 0.9
e = 0.0
w = None
iht = None
z = None
y = 1
last_state = None
last_action = None
position = None
velocity = None
actionResult = None
iht = IHT(memorySize)


def agent_init():
	global w,z,iht,position,velocity
	randomNumber = random.uniform(-0.001,0.0)
	w = np.full(memorySize,randomNumber)
	# Initialize parameter vector w
	
	z = np.zeros(memorySize)
	position = num_tilings / (max_position - min_position)
	velocity = num_tilings / ( max_velocity - min_velocity)	
	return

def agent_start(state):
	global actions,iht,position,velocity,z,memorySize,e,num_tilings,last_action,last_state
	
	
	if np.random.binomial(1, e) == 1:
		action =  np.random.choice(actions)
	actionResult = []
	for a in actions:
		targetTiles = tiles(iht, num_tilings,[state[0]*position , state[1]*velocity],[a])
		results = np.sum(w[targetTiles])
		actionResult.append(results)
	action = actions[np.argmax(actionResult)]
	# Choose A  or "-greedy according to q
	
	last_state = state
	last_action = action
	
	z = np.zeros(memorySize)
	#z <-0
	
	return action


def agent_step(reward, state): 
	global actions,iht,position,velocity,z,memorySize,e,num_tilings,last_action,last_state,w,y
	
	
	delta = reward
	# delta <- R
	last_Tiles = tiles(iht, num_tilings,[position * last_state[0] , velocity * last_state[1]],[last_action])
	
	for n in last_Tiles:
		delta = delta - w[n]
		z[n] = 1
	# Loop for i in F(S, A):
	#               delta = delta - w[i]
	#               z_i = 1
	
	
	if np.random.binomial(1, e) == 1:
		action =  np.random.choice(actions)
	actionResult = []
	for a in actions:
		targetTiles = tiles(iht, num_tilings,[state[0]*position , state[1]*velocity],[a])
		results = np.sum(w[targetTiles])
		actionResult.append(results)
	action = actions[np.argmax(actionResult)]
	#ChooseA orneargreedily q
	
	targetTiles = tiles(iht, num_tilings,[state[0]*position , state[1]*velocity],[action])
	for n in targetTiles:
		delta +=y*w[n]
	#Loop for i in F(S0,A0):    
	#         delta = delta+ wi
		
	w += alpha * delta * z
	# w = w + delta * z
	
	z *= lada * y
	# z = z * landa * y
	
	last_action = action
	last_state = state
	
	
	return action
def agent_end(reward):
	global actions,iht,position,velocity,z,memorySize,e,num_tilings,last_action,last_state,w,y

	delta = reward
	last_Tiles = tiles(iht, num_tilings,[position * last_state[0] , velocity * last_state[1]],[last_action])
	for n in last_Tiles:
		delta = delta - w[n]
		z[n] = 1
		
	w += alpha * delta * z
	
	
	return 
def agent_cleanup():

	return

def agent_message(in_message): # returns string, in_message: string
   	
	if (in_message == 'ValueFunction'):
		return v
	else:
		return "I don't know what to return!!"
    


