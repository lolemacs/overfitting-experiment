from os import listdir
from os.path import isfile, join
import numpy as np
import cPickle

fileNames = [f for f in listdir(".") if isfile(join(".", f))]
Z = []
for fileName in fileNames:
    if ".pkl" in fileName:
        print fileName
        with open(fileName,"rb") as f:
            Z.append(cPickle.load(f))
Z = np.array(Z)


print Z.shape
