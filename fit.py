import generate
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score

h2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
h10 = make_pipeline(PolynomialFeatures(10), LinearRegression())

nIters = (120-20)/5 * (200-0)/5
i = 0

x = np.arange(6)
y = np.arange(5)
print y[:, np.newaxis]
quit()

Z = []

#first experiment, Q fixed as 20
for N in range(20,120,5):
    row = []
    for sigma in range(0,200,5):
        data = generate.generate(20, sigma/100., N)
        X = data[0].reshape(N,1)
        Y = data[1].reshape(N,1)
        h2.fit(X, Y)
        h10.fit(X, Y)
        err2 = ((h2.predict(X) - Y)**2).mean()
        err10 = ((h10.predict(X) - Y)**2).mean()
        i += 1
        if i in range(0, nIters, nIters/10): print float(i)/nIters
        #print "Err2(in), Err10(in) (%s,%s): " % (N,sigma), err2, err10
        row.append(err2 - err10)
    Z.append(row)

plt.figure(1)
plt.imshow(Z)
plt.clim()
plt.title("Boring slide show")
plt.show()
