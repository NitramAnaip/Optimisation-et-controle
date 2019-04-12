import numpy as np
from Probleme_R import *
from Structures_N import *

def OraclePG(qc, ind = 4):
    if ind == 2:
        q = q0 + np.dot(B,qc)
        val = np.dot(q,r*q*abs(q))/3 + np.dot(pr,np.dot(Ar,q))
        return val
    if ind == 3:
        q = q0 + np.dot(B,qc)
        grad = np.zeros(n-md)
        prod = r*q*np.abs(q) + np.dot(np.transpose(Ar),pr)
        grad = np.dot(np.transpose(B),prod)

        return grad
    if ind == 4:
        q = q0 + np.dot(B,qc)

        val = np.dot(q,r*q*abs(q))/3 + np.dot(pr,np.dot(Ar,q))

        grad = np.zeros(n-md)
        prod = r*q*np.abs(q) + np.dot(np.transpose(Ar),pr)
        grad = np.dot(np.transpose(B),prod)

        return val, grad
