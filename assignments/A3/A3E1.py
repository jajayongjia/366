


import numpy as np
import matplotlib.pyplot as plt

THETA = 0.0000000000000000000000000000001 
num_capital = 100
a = 1


for ph in [0.4,0.25,0.55]:
    
    states = np.arange(num_capital+1) # states list contains 0.....100 where 0 and 100 are terminal states.
    
    V = np.zeros(num_capital + 1) # all states including starting state and final state
    V[num_capital] = 1 # the terminal condition
    policys = np.zeros(num_capital-1)   
    sweep = 0
    Vfinal = np.zeros((99,4))
    while True:
        
        delta = 0.0
        for s in states[1:num_capital]:   #for each states except the terminal states
    
            v = V[s]
            actions = np.arange(min(s, num_capital - s) + 1) # a = {0, 1, . . . , min(s, 100   s)}.
            returnValue = []
            for action in actions[1:]: 
                if action+s >=100:
                    value = ph * (1+V[s+action]) + (1.0-ph) * (0+V[s-action])
                else:
                    value = ph * (V[s+action]) + (1.0-ph) * (V[s-action])
                returnValue.append(value)
		
            V[s] = np.max(returnValue)
            index = returnValue.index(V[s])
            bestAction = actions[index]
            policys[s-1] = bestAction
            # V(s) maxa of (p(s,r|s,a)[ r+ V(s')]
            delta = np.max([delta, np.abs(v-V[s])])    
	    if sweep<3:
		    Vfinal[s-1][sweep] = V[s]
	    else:
		    Vfinal[s-1][3] = V[s]            
            
        if delta< THETA :
            break
            # Terminal condition,
        sweep+=1
        
    plt.figure(a)
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.plot(Vfinal[1:num_capital])
    a +=1
    plt.figure(a)
    a+=1
    plt.plot(states[1:num_capital], policys)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
               
               
    plt.show()

