import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	result= open("result.txt", "r")
	steps = []
	for pairs in result:
		pairs_n = pairs.split("--")
		for data in pairs_n[:3]:
			a = data.split(",")
			a = a[:-1]
			steps.append(a)

        for n in steps:
		plt.plot(n)		
	plt.xlim(1,50)
	plt.ylim(1,800)
	plt.xticks([0,10,20,30,40,50])
	plt.yticks([0,200,400,600,800])	
	plt.ylabel('steps per episode')
	plt.xlabel('episode')
	plt.show() 