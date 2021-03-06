import numpy as np
from Probleme_R import *
from Structures_N import *

def OracleDG(Lambda, ind = 4):
    q = np.sign(- np.dot(np.transpose(Ar),pr) - np.dot(np.transpose(Ad),Lambda)) \
            * np.sqrt(np.absolute((np.dot(np.transpose(Ar),pr) + np.dot(np.transpose(Ad),Lambda))/r))

    if ind == 2:
        val = np.dot(q,r*q*np.abs(q))/3 + np.dot(pr,np.dot(Ar,q)) + np.dot(Lambda, np.dot(Ad, q) - fd)
        return -val

    if ind == 3:
        grad = np.dot(Ad, q) - fd
        return -grad

    if ind == 4:
        val = np.dot(q,r*q*np.abs(q))/3 + np.dot(pr,np.dot(Ar,q)) + np.dot(Lambda, np.dot(Ad, q) - fd)
        grad = np.dot(Ad, q) - fd

        return -val, -grad
