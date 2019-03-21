#!/usr/bin/python

import numpy as np
from Probleme_R import *
from Structures_N import *

def OraclePH(qc, ind = 7):
    q = q0 + np.dot(B,qc)
    if ind == 2:
        val = np.dot(q,r*q*abs(q)) + np.dot(pr,np.dot(Ar,q))
        return val
    if ind == 3:
        grad = np.zeros(n-md)
        prod = r*q*np.abs(q) + np.dot(np.transpose(Ar),pr)
        grad = np.dot(np.transpose(B),prod)
        return grad
    if ind == 4:
        val = np.dot(q,r*q*abs(q)) + np.dot(pr,np.dot(Ar,q))
        grad = np.zeros(n-md)
        prod = r*q*np.abs(q) + np.dot(np.transpose(Ar),pr)
        grad = np.dot(np.transpose(B),prod)
        return val, grad
    if ind == 5:
        prod2 = r*np.abs(q)
        H = 2*np.dot(np.transpose(B),np.dot(np.diag(prod2),B))
        return H
    if ind == 6:
        grad = np.zeros(n-md)
        prod = r*q*np.abs(q) + np.dot(np.transpose(Ar),pr)
        prod2 = r*np.abs(q)
        grad = np.dot(np.transpose(B),prod)
        H = 2*np.dot(np.transpose(B),np.dot(np.diag(prod2),B))
        return grad, H
    if ind == 7:
        val = np.dot(q,r*q*abs(q)) + np.dot(pr,np.dot(Ar,q))
        grad = np.zeros(n-md)
        prod = r*q*np.abs(q) + np.dot(np.transpose(Ar),pr)
        prod2 = r*np.abs(q)
        grad = np.dot(np.transpose(B),prod)
        H = 2*np.dot(np.transpose(B),np.dot(np.diag(prod2),B))
        return val, grad, H
