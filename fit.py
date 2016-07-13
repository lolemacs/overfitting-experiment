import generate
import numpy as np
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score

linreg = linear_model.LinearRegression()

for N in range(20,120,5):
    for sigma in range(0,200,5):
        data = generate.generate(20, sigma/100., N)
        linreg.fit([data[0]], [data[1]])
        print "Err(in): ", linreg.score(data[0], data[1])
