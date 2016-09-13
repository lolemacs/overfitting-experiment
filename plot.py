from os import listdir
from os.path import isfile, join
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from matplotlib import cm

fileNames = [f for f in listdir(".") if isfile(join(".", f))]
Z = []
for fileName in fileNames:
    if ".pkl" in fileName:
        print fileName
        with open(fileName,"rb") as f:
            if Z == []: Z = cPickle.load(f)
            else: Z = np.concatenate((Z,cPickle.load(f)),axis=0)
print Z.shape

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
