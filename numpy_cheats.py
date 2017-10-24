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
    #print(arr_of_zip)

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

    'load txt'
    siteA = np.loadtxt('data/siteA.txt')


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

def creation_2d():
    '2d Constants'
    arr1 = np.zeros(shape=(10,3))
    arr2 = np.ones(shape=(10,3))

    'reshape'
    arr3 = np.zeros(30).reshape(10,3)

    'load txt'
    twoDdata = np.loadtxt('data/some_data.txt', skiprows=1)
    print(twoDdata)

    'concatenation'
    arr1 = np.zeros(10).reshape(10,1)
    arr2 = np.ones(10).reshape(10,1)
    arr3 = np.concatenate((arr1, arr2), axis = 1)
    print(arr3)

    'from pandas'

def get_2_3_elements(ell):
    return ell[1:3]

def concatenate_column(X):
    n = X.shape[0]
    ones = np.ones(shape=(n,1))
    X = np.concatenate([X,ones],axis=1)
    return X

def elementwise_ops():

    'vectorize function'
    f = get_2_3_elements
    fv = np.vectorize(f)
    arr = np.array(['hello','folks'])
    fv(arr)
    #--> ['el','or']

    'Elementwise broadcasting'
    '--Scalars'
    arr = np.array([[1,1],[5,10]])
    a = 10
    print(a-arr)
    print(a+arr)
    print(a*arr)
    print(arr/a)
    print(a/arr)

    '--Matching Dimensions'
    arr2 = np.array([[3,4],[3,-2]])
    print(arr+arr2)
    print(arr-arr2)
    print(arr*arr2)
    print(arr/arr2)
    print(arr2/arr)


def row_column_wise_ops():
    arr = np.array([[1,1],[5,10]])
    #print(arr)
    arr=concatenate_column(arr)
    #print(arr)

    'axis = 0 doesnt mean across rows. It means "with increasing row index"'
    'i.e. down. Usually crushing the columns. Sort of "along each column"  '

    'axis = 1 means "along each row." with increasing column index'

    print(arr.mean(axis=0))
    print(arr.mean(axis=1))

def linear_algebra():
    pass



if __name__ == "__main__":
    #creation_1d()
    #creation_2d()
    #elementwise_ops()
    row_column_wise_ops()
