import generate
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm

h2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
h10 = make_pipeline(PolynomialFeatures(10), LinearRegression())

nIters = (120-20)/5 * (200-0)/5

dump = []
debug = False

reps = 30
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
                err2 = ((np.clip(h2.predict(testX),-10,10) - testY)**2).mean()
                err10 = ((np.clip(h10.predict(testX),-10,10) - testY)**2).mean()

            dump.append(err10)

            i += 1
            if i in range(0, nIters, nIters/10): print float(i)/nIters
            row.append(err10 - err2)
        z.insert(0,row)
    z = np.clip(z, -2.0, 2.0)
    Z.append(z)

Z = np.array(Z)

#print Z.max(), Z.min()

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
