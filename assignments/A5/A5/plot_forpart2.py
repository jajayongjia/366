import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	result= open("result2.txt", "r")
	steps = []
	aList = [0.03125,0.0625,0.125,0.25,0.5,1.0]
	for pairs in result:
		pairs_n = pairs.split("--")
		for data in pairs_n[:6]:
			a = data.split(",")
			a = a[:-1]
			steps.append(float(a[0]))

	plt.plot(aList,steps)	
	plt.ylim(1,100)
	plt.xticks([0.03125,0.0625,0.125,0.25,0.5,1.0])
	plt.yticks([0,20,40,60,80,100])	
	plt.ylabel('steps per episode')
	plt.xlabel('alpha')
	plt.show() 
