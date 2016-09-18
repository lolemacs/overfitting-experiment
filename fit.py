import generate
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
import cPickle
import random
import time

h2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
h10 = make_pipeline(PolynomialFeatures(10), LinearRegression())

nIters = (120-20)/5 * (200-0)/5

reps = 2000
step = 5
nTestSamples = 200

Z = []
for rep in range(reps):
    i = 0
    z = []
    for var in range(0,200,step):
        row = []
        for N in range(20,120,step):
            X, Y, legPol = generate.generate(20, var/100., N + nTestSamples)

            X = X.reshape(-1,1)
            Y = Y.reshape(-1,1)

            trainX, testX = X[:N], X[N:]
            trainY, testY = Y[:N], Y[N:]

            h2.fit(trainX, trainY)
            h10.fit(trainX, trainY)

            err2 = ((h2.predict(testX) - testY)**2).mean()
            err10 = ((h10.predict(testX) - testY)**2).mean()
 
            i += 1
            if i in range(0, nIters, nIters/10): print "%s: %s"%(rep,float(i)/nIters)
            row.append(err10 - err2)
        z.insert(0,row)
    Z.append(z)

Z = np.array(Z)

with open("%s-%s.pkl"%(random.randint(0,9999999),Z.shape[0]),"wb") as f:
    cPickle.dump(Z,f)
