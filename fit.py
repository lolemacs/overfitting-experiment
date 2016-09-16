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

dump = []
debug = False

reps = 5
step = 5
nTestSamples = 200

Z = []
for rep in range(reps):
    i = 0
    z = []
    #first experiment, Q fixed as 20
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

            #plt.figure(1)
            #x = np.linspace(-1.,1.,100)

            #plt.plot(x, legPol(x))
            #plt.plot(x, h2.predict(x.reshape(-1,1)))
            #plt.plot(x, h10.predict(x.reshape(-1,1)))
            #plt.show()
            
            if debug:
                err2 = ((h2.predict(trainX) - trainY)**2).mean()
                err10 = ((h10.predict(trainX) - trainY)**2).mean()

            else:
                err2 = ((h2.predict(testX) - testY)**2).mean()
                err10 = ((h10.predict(testX) - testY)**2).mean()
 
            def E(p):
                integ = p.integ()
                return (integ(1) - integ(-1))/2

            #print E(legPol)

            #coefs2 = h2.named_steps['linearregression'].coef_
            #coefs10 = h10.named_steps['linearregression'].coef_
            #fit2 = np.polynomial.Polynomial(coefs2[0])
            #fit10 = np.polynomial.Polynomial(coefs10[0])
            
            #aerr2 = E(fit2**2) + 2 * E(fit2) * E(legPol) + E(legPol**2)
            #aerr10 = E(fit10**2) + 2 * E(fit10) * E(legPol) + E(legPol**2)

            #print "Analytical error: ", aerr10 - aerr2
            #print "Empirical error: ", err10 - err2
            #dump.append(err10)

            i += 1
            if i in range(0, nIters, nIters/10): print "%s: %s"%(rep,float(i)/nIters)
            row.append(err10 - err2)
        z.insert(0,row)
    #z = np.clip(z, -2.0, 2.0)
    Z.append(z)

Z = np.array(Z)

#quit()

with open("%s-%s.pkl"%(random.randint(0,9999999),Z.shape[0]),"wb") as f:
    cPickle.dump(Z,f)

Z = Z.mean(axis=0)

#print Z

Z = np.clip(Z, -.2, .2)
#print max(dump)

fig, ax = plt.subplots()
cax = ax.imshow(Z, interpolation='bicubic', cmap=cm.inferno)
ax.set_title("Things")
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['0'])
#plt.xlabel('Sigma')
#plt.ylabel('N')
plt.show()
