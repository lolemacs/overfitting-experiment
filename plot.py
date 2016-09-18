from os import listdir
from os.path import isfile, join
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from matplotlib import cm

fileNames = [f for f in listdir(".") if isfile(join(".", f))]
Z = []
for fileName in fileNames:
    if ".pkl" in fileName and fileName[0] != "_":
        print fileName
        with open(fileName,"rb") as f:
            if Z == []: Z = cPickle.load(f)
            else: Z = np.concatenate((Z,cPickle.load(f)),axis=0)
print Z.shape

#Z = Z.mean(axis=0)
Z = np.median(Z,axis=0)

#print Z

Z = np.clip(Z, -.2, .2)
#print max(dump)

fig, ax = plt.subplots()
cax = ax.imshow(Z, interpolation='bicubic', cmap=cm.inferno, extent=[20,120,0,200])
ax.set_title("Stochastic Noise")
ax.set_xlabel("Number of data points")
ax.set_ylabel("Noise x 100")
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['0'])
#plt.xlabel('Sigma')
#plt.ylabel('N')
plt.show()


Z = []
for fileName in fileNames:
    if ".pkl" in fileName and fileName[0] == "_":
        print fileName
        with open(fileName,"rb") as f:
            if Z == []: Z = cPickle.load(f)
            else: Z = np.concatenate((Z,cPickle.load(f)),axis=0)
print Z.shape

Z = Z.mean(axis=0)
#Z = np.median(Z,axis=0)

#print Z

Z = np.clip(Z, -.2, .2)
#print max(dump)

fig, ax = plt.subplots()
cax = ax.imshow(Z, interpolation='bicubic', cmap=cm.inferno, extent=[20,120,0,100])
ax.set_title("Deterministic Noise")
ax.set_xlabel("Number of data points")
ax.set_ylabel("Target Complexity")
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['0'])
#plt.xlabel('Sigma')
#plt.ylabel('N')
plt.show()
