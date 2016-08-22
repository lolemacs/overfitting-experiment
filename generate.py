import math
import numpy as np
import numpy.polynomial.legendre as leg

def L(x, k):
    #old
    l = [1, x]
    for i in range(2, k):
        l.append( (2.*i - 1)/i * x * l[-1] - (i - 1.0)/i * l[-2] )
    return np.array(l)

def get_coefs(k):
    #need to return coefficients that normalize the polynomio
    return np.random.normal(size=k)

def f(x, k):
    #old
    coefs = get_coefs(k)
    Lq = L(x,k)
    return np.sum(coefs * Lq)

def noise(sigma, n):
    #return gaussian noise with sigma std
    return np.random.normal(size=n) * sigma
    
def generate(Q, sigma, N):
    #generates N random points in the [-1,+1] interval
    X = np.random.uniform(low=-1, high = 1, size = N)

    #get random coefficients and generates the Legendre polynomio
    coefs = get_coefs(Q+1)
    legPol = leg.Legendre(coefs)

    #calculates target values for each x \in X (also adding noise)
    Y = map(lambda x: legPol(x), X) + noise(sigma, N)
    return [X,Y]
