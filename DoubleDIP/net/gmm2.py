import os
#import h5py
import numpy as np
import torch
import cv2
import random
import skimage.measure

# GMM initialization
def initialize(X,k):
    n = X.shape[1]
    idx = random.sample(range(n),k)
    m = X[:,idx]
    a = m.T*X-np.square(m).T/2
    #R = np.argmax(a,0)
    R = list(map(lambda x: x==np.max(x,0),a.T)) * np.ones(shape=a.T.shape)
    return R

# E step
def expectation(X,model):
    mu = model['mu']
    Sigma = model['Sigma']
    w = model['weight']

    n = X.shape[1]
    k = mu.shape[1]
    logRho = np.zeros([n,k])

    for i in range(k):
        logRho[:,i] = loggausspdf(X,mu[0,i],Sigma[0,i])

    logRho = logRho+np.log(w)
    T = logsumexp(logRho)
    llh = np.sum(T)/n
    logR = logRho.T-T
    R = np.exp(logR).T
    return (R,llh)

def loggausspdf(X,mu,Sigma):
    d = X.shape[0]
    X = X-mu
    U = np.sqrt(Sigma)
    Q = X/U
    q = np.square(Q)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(U))
    y = -(c+q)/2
    return y

def logsumexp(x):
    y = np.max(x,1)
    x = x.T-y
    s = y+np.log(np.sum(np.exp(x),0))
    return s

# M step
def maximizationModel(X,R):
    k = R.shape[1]
    nk = np.sum(R,0)
    mu = np.zeros([1,k])

    w = nk/R.shape[0]
    Sigma = np.zeros([1,k])
    sqrtR = np.sqrt(R)
    for i in range(k):
        Xo = X-mu[0,i]
        Xo = Xo*sqrtR[:,i]
        Sigma[:,i] = np.dot(Xo,Xo.T)/nk[i]
        Sigma[:,i] = Sigma[:,i]+1e-6

    model = {'mu':mu, 'Sigma':Sigma, 'weight':w}
    return model



