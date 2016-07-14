import generate
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score

h2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
h10 = make_pipeline(PolynomialFeatures(10), LinearRegression())

for N in range(20,120,5):
    for sigma in range(0,200,5):
        data = generate.generate(20, sigma/100., N)
        X = data[0].reshape(N,1)
        Y = data[1].reshape(N,1)
        h2.fit(X, Y)
        h10.fit(X, Y)
        err2 = ((h2.predict(X) - Y)**2).mean()
        err10 = ((h10.predict(X) - Y)**2).mean()
        print "Err2(in) - Err10(in) (%s,%s): "%(N,sigma), err2 - err10
