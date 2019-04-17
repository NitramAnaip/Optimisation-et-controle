#!/usr/bin/python

import numpy as np
from Probleme_R import *
from Structures_N import *

def OracleDH(qc, ind = 7):
    q = np.sign(-np.dot(np.transpose(Ar),pr) - np.dot(np.transpose(Ad),Lambda)) \
            * np.sqrt(np.absolute((np.dot(np.transpose(Ar),pr) + np.dot(np.transpose(Ad),Lambda))/r)) ## ATTENTION: vérifier que diviser par r divise bien element wise

    if ind == 2:
        val = np.dot(q,r*q*abs(q))/3 + np.dot(pr,np.dot(Ar,q)) + np.dot(Lambda, np.dot(Ad, q) - fd)
        return val
    if ind == 3:
        grad = np.dot(Ad, q) - fd
        return grad
    if ind == 4:
        val = np.dot(q,r*q*abs(q))/3 + np.dot(pr,np.dot(Ar,q)) + np.dot(Lambda, np.dot(Ad, q) - fd)
        grad = np.dot(Ad, q) - fd
        return val, grad
    if ind == 5:
        ## RESTE LA HESSIENNE A CODER
        return H
    if ind == 6:
        grad = np.dot(Ad, q) - fd
        return grad, H
    if ind == 7:
        val = np.dot(q,r*q*abs(q))/3 + np.dot(pr,np.dot(Ar,q)) + np.dot(Lambda, np.dot(Ad, q) - fd)
        grad = np.dot(Ad, q) - fd
        return val, grad, H
