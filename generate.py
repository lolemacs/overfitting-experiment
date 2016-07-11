import math
import numpy as np

def L(x, k):
    l = [1, x]
    for i in range(2, k):
        l.append( (2.*i - 1)/i * x * l[-1] - (i - 1.0)/i * l[-2] )
    return np.array(l)

def get_coefs(k):
    return np.random.normal(size=k)

def f(x, k):
    coefs = get_coefs(k)
    Lq = L(x,k)
    return np.sum(coefs * Lq)

def noise(sigma, n):
    return np.random.normal(size=n) * sigma
    
def generate(Q, sigma, N):
    X = np.random.uniform(low=-1, high = 1, size = N)
    Y = map(lambda x: f(x,Q), X) + noise(sigma, N)
    return [X,Y]

for N in range(20,120,5):
    for sigma in range(0,200,5):
        generate(20,sigma/100.,N)
