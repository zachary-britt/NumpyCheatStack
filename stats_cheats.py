import numpy as np
import scipy.stats as scs

def neg_s(v):
    if v<0:
        return ""
    else:
        return " "


def hypothesis_testing_one_sample(X, H0_mu, alpha=0.05):
    SE = scs.sem(X)
    mu = X.mean()
    n = len(X)
    df = n-1

    t_stat = -abs(mu-H0_mu)/SE

    dist = scs.t(df)
    p = dist.cdf(t_stat)


    print("T score:\t {}{:.2}".format(neg_s(t_stat) , t_stat))
    print("p val:\t\t {}{:.2}".format(neg_s(t_stat) , p))


if __name__ == "__main__":
    n = 20
    H1_mu = 110
    H0_mu = 100
    sig = 10
    X1 = np.random.normal(loc = H1_mu, scale = sig, size=n)
    X0 = np.random.normal(loc = H0_mu, scale = sig, size=n)

    alpha = 0.05

    hypothesis_testing_one_sample(X1, H0_mu, alpha)
