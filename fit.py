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
i = 0

Z = []

dump = []
debug = False

#first experiment, Q fixed as 20
for N in range(20,120,5):
    row = []
    for sigma in range(0,200,5):
        X, Y, legPol = generate.generate(20, sigma/100., N*2)

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

        dump.append(err10)
        #if err10 > 5: print err10

        i += 1
        if i in range(0, nIters, nIters/10): print float(i)/nIters
        #print "Err2(in), Err10(in) (%s,%s): " % (N,sigma), err2, err10
        row.append(err2 - err10)
    Z.append(row)

#print max(dump)

fig, ax = plt.subplots()
cax = ax.imshow(Z, interpolation='bicubic', cmap=cm.coolwarm)
ax.set_title("Boring slide show")
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
#plt.xlabel('Sigma')
#plt.ylabel('N')
plt.show()
