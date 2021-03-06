#!/usr/bin/python

import numpy as np
from numpy import dot

########################################################################
#                                                                      #
#          RECHERCHE LINEAIRE SUIVANT LES CONDITIONS DE WOLFE          #
#                                                                      #
#          Algorithme de Fletcher-Lemarechal                           #
#                                                                      #
########################################################################

#  Arguments en entree
#
#    alpha  : valeur initiale du pas
#    x      : valeur initiale des variables
#    D      : direction de descente
#    Oracle : nom de la fonction Oracle
#
#  Arguments en sortie
#
#    alphan : valeur du pas apres recherche lineaire
#    ok     : indicateur de reussite de la recherche
#             = 1 : conditions de Wolfe verifiees
#             = 2 : indistinguabilite des iteres

def Wolfe(alpha, x, D, Oracle):

    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf
    M = 10**20

    ok = 0
    dltx = 0.00000001

    ##### Algorithme de Fletcher-Lemarechal

    # Appel de l'oracle au point initial
    critere, gradient = Oracle(x)

    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x

    # Boucle de calcul du pas
    while ok == 0:

        # xn represente le point pour la valeur courante du pas,
        # xp represente le point pour la valeur precedente du pas.
        xp = xn
        xn = x + alpha_n * D

        # Calcul des conditions de Wolfe

        critere2, gradient2 = Oracle(xn)
        condition_1 = (critere2 <= critere + omega_1 * alpha_n * np.dot(np.transpose(gradient),D))

        # Test des conditions de Wolfe
        # - si les deux conditions de Wolfe sont verifiees,
        #   faire ok = 1 : on sort alors de la boucle while
        # - sinon, modifier la valeur de alphan : on reboucle.

        if not(condition_1):
            alpha_max = alpha_n
            alpha_n = (alpha_max + alpha_min) /2
        else:
            condition_2 = (np.dot(np.transpose(gradient2),D) >= omega_2 * np.dot(np.transpose(gradient),D))
            if not(condition_2):
                alpha_min = alpha_n
                if alpha_max > M:
                    alpha_n = 2*alpha_min
                else:
                    alpha_n = (alpha_max + alpha_min) /2
            else:
                ok = 1

        # Test d'indistinguabilite
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok
