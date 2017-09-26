import numpy as np
import numpy.linalg as npl
import pandas as pd
import scipy.stats as scs

def creation_1d():
    n = 20

    'Constants'
    arr_ones = np.ones(n)
    arr_zeros = np.zeros(n)

    'Stepped Vectors'
    arr_steps = np.linspace(0,20, 21) # 201 points in [0,20]
    arr_steps2= np.arange(0,21,1) # 1 sized steps in [0,21)

    'from list or listlike'
    l1 =[4,1,0,10]
    l2 =['a','b','c','d']
    arr_of_list = np.array(l1)
    arr_of_range= np.array(range(1,20))
    arr_of_zip = np.array(list(zip(l1,l2)))
    print(arr_of_zip)

    'RVs'
    arr_rand_uniform = np.random.rand(n) # n uniforms (0,1)
    arr_rand_uniform2 = np.random.random(n)
    arr_rand_int = np.random.randint(low=0,high=100,size=n) #random int in [0,100)
    arr_rand_norm = np.random.randn(n) # n standard normals

    'Sample Draw'
    sample = arr_rand_int
    probs = arr_rand_uniform / sum(arr_rand_uniform)
    #draw from urn
    S = np.random.choice(sample, size=n, replace=True)
    S2 = np.random.choice(sample,size=n, replace=True, p=probs)

    '''
    See also:
    np.random.beta
    np.random.binomial
    np.random.chisquare
    np.random.exponential
    np.random.gamma
    np.random.geometric
    np.random.hypergeometric
    np.random.logistic
    np.random.multivariate_normal
    np.random.normal
    np.random.poisson
    '''



if __name__ == "__main__":
    creation_1d()
