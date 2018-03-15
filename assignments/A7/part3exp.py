#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "part3agent")
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    num_episodes = 1000
    
    RL_init()
    for e in range(num_episodes):
        print '\tepisode {}'.format(e+1)
        RL_episode(0)
    
    (x,y,z) = RL_agent_message("HEIGHT")
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    # reference : https://stackoverflow.com/questions/2294588/python-how-to-plot-3d-graphs-using-python
    ax.scatter(x, y, z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go') 
    plt.show()