#!/usr/bin/python

import numpy as np

#############################################################################
#                                                                           #
#  MONITEUR D'ENCHAINEMENT POUR LE CALCUL DE L'EQUILIBRE D'UN RESEAU D'EAU  #
#                                                                           #
#############################################################################

##### Fonctions fournies dans le cadre du projet

# Donnees du probleme
from Probleme_R import *
from Structures_N import *

# Affichage des resultats
from Visualg import Visualg

# Verification des resultats
from HydrauliqueP import HydrauliqueP
from HydrauliqueD import HydrauliqueD
from Verification import Verification

# Oracles pour le problème primal et le problème dual
from OraclePG import OraclePG
from OraclePH import OraclePH

from OracleDG import OracleDG
from OracleDH import OracleDH

# Exemple 1 - le gradient a pas fixe :
from Gradient_F import Gradient_F

# Exemple 2 - Minimisations a pas variable
from Minimize import Gradient_V, Polak_Ribiere, BFGS

# Exemple 3 - Algorithme de Newton
from Newton_F import Newton_F

##### Initialisation de l'algorithme

# ---> La dimension du vecteur dans l'espace primal est n-md
#      et la dimension du vecteur dans l'espace dual est md


# Probleme primal :
# x0 = 0.1 * np.random.normal(size=n-md)

# Probleme dual :
x0 = 100 + np.random.normal(size=md)

##### Minimisation proprement dite
# print()
# print("ALGORITHME DU GRADIENT A PAS FIXE")
# copt, gopt, xopt = Gradient_F(OraclePG, x0) # Cas primal
# copt, gopt, xopt = Gradient_F(OracleDG, x0, gradient_step=0.6) # Cas dual

# print()
# print("NEWTON A PAS FIXE")
# copt, gopt, xopt = Newton_F(OraclePH, x0)
copt, gopt, xopt = Newton_F(OracleDH, x0)


# Exemple 2 - le gradient a pas variable :
#
# # print()
# print("ALGORITHME DU GRADIENT A PAS VARIABLE")
# copt, gopt, xopt = Gradient_V(OraclePG, x0)
# copt, gopt, xopt = Gradient_V(OracleDG, x0)

# print()
# print("ALGORITHME DE POLAK RIBIERE")
# copt, gopt, xopt = Polak_Ribiere(OraclePG, x0)
# copt, gopt, xopt = Polak_Ribiere(OracleDG, x0, gradient_step_ini=15)

#
# print()
# print("ALGORITHME BFGS")
# copt, gopt, xopt = BFGS(OraclePG, x0)
# copt, gopt, xopt = BFGS(OracleDG, x0, gradient_step_ini=20)

##### Verification des resultats

# ---> La fonction qui reconstitue les variables hydrauliques
#      du reseau a partir de la solution du probleme s'appelle
#      HydrauliqueP pour le probleme primal, et HydrauliqueD
#      pour le probleme dual
#
# Probleme primal :
# qopt, zopt, fopt, popt = HydrauliqueP(xopt)

# Probleme dual :
qopt, zopt, fopt, popt = HydrauliqueD(xopt)

Verification(qopt, zopt, fopt, popt)
