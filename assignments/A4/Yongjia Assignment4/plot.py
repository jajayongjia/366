#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	result= open("result.txt", "r")
	steps = []
	episodes = []
	#plt.show()
	for pairs in result:
		data = pairs.split(",")
		steps.append(data[0])
		episodes.append(data[1])	
	for testPoints in [39,79,119,159]:	
		plt.plot(steps[testPoints],episodes[testPoints],"ro")
		plt.text(steps[testPoints],episodes[testPoints], str(steps[testPoints])+","+str(episodes[testPoints]))
	plt.plot(steps,episodes)
	plt.xlabel('steps')
	plt.ylabel('episode')	
	plt.show()