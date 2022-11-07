#%%
import plotly.express as px
import pandas as pd
import numpy as np
import json
from timeit import default_timer as timer
from numba import jit


@jit(nopython=True)
def add(X):
	s = 0.0
	for i in X:
		s += i
	return s


@jit(nopython=True)
def MSE(X,Y,theta):
	m = len(Y)
	return 1/m * add(np.square(Y - np.dot(X,theta)))

@jit(nopython=True)
def BGD(X,Y,theta):
	m = len(Y)
	a = 0.00001
	iterations = 1500
	J = [0 for _ in range(iterations)]
	for i in range(iterations):
		theta = theta - a * 2/m * np.dot(X.T, (np.dot(X,theta) - Y))
		J[i] = MSE(X,Y,theta)

	return theta,J

@jit(nopython=True)
def NormalEQ(X,Y):
		return np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,Y))



df = pd.read_csv("Advertising.csv")
df_m = df.to_numpy().T

X = df_m[0:4].T
X.T[0] = [1.0 for _ in range(len(X))]
Y = df_m[-1]
theta = np.array([0.0 for i in range(len(X.T))])
t = np.array([0.0 for i in range(len(X.T))])
time = {
	"GD":0.0,
	"NE":0.0
}
NormalEQ(X,Y)

print("Initial Costs:",MSE(X,Y,theta))

for _ in range(100):
	start = timer()
	theta,J = BGD(X,Y,theta)
	end = timer()
	time["GD"] += end - start
	#print("Final Costs:",MSE(X,Y,theta))

	start = timer()
	theta = NormalEQ(X,Y)
	end = timer()
	time["NE"] += end - start
	print(end - start)

time["GD"] /= 100
time["NE"] /= 100


print("Final Costs:",MSE(X,Y,theta))

with open("pyLR_numba_output.json", "w") as f:
	json.dump(time,f)

