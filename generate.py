import math
import numpy as np
import numpy.polynomial.legendre as leg
import matplotlib.pyplot as plt

def get_coefs(k):
    #return coefficients sampled from a normal distribution
    return np.random.normal(size=k)

def noise(sigma, n):
    #return gaussian noise with sigma std
    return np.random.randn(n) * sigma

def genNormalizedLeg(coefs):
    #generates a legendre polynomio with given coefficients
    legPol = leg.Legendre(coefs)

    #calculates the integral of the squared polynomio
    I = (legPol**2).integ()

    #calculates the norm and normalizes the polynomio accordingly
    norm = (I(1) - I(-1))/2
    legPol /= math.sqrt(norm)
    return legPol

def generate(Q, var, N):
    #generates N random points in the [-1,+1] interval
    X = np.random.uniform(low=-1, high = 1, size = N)

    #get random coefficients and generates the Legendre polynomio
    coefs = get_coefs(Q+1)

    #generates a normalized Legendre polynomio
    legPol = genNormalizedLeg(coefs)

    #calculates target values for each x \in X (also adding noise)
    Y = map(lambda x: legPol(x), X) + noise(math.sqrt(var), N)

    return [X,Y, legPol]

if __name__ == "__main__":
    X, Y, legPol = generate(20, .0, 10000)
    print (Y**2).mean()
    plt.figure(1)
    x = np.linspace(-1.,1.,100)
    plt.plot(x, legPol(x))
    plt.show()
