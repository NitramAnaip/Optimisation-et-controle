import numpy as np
from time import process_time
import copy
from Wolfe_Skel import Wolfe
from Visualg import Visualg

def Polak_Ribiere(Oracle, x0):

    ##### Initialisation des variables

    iter_max = 10000
    gradient_step = 0.0005
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0



    ##### Boucle sur les iterations

    for k in range(iter_max):

        # Valeur du critere et du gradient + calcul du beta
        critere, gradient = Oracle(x)
        if k==0:
            D = -gradient
            beta = 0
        else:
            diff = gradient-gradient_ant
            numerator = np.dot(np.transpose(diff),gradient)
            norm = np.linalg.norm(gradient_ant)**2
            beta = numerator/norm

        # Test de convergence
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        D = -gradient + beta*D

        # Mise a jour des variables
        gradient_ant = copy.copy(gradient)
        gradient_step, ok = Wolfe(gradient_step, x, D, Oracle)
        x = x + (gradient_step*D)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

    ##### Resultats de l'optimisation

    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', np.linalg.norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt

def Gradient_V(Oracle, x0):

    ##### Initialisation des variables
    iter_max = 10000
    gradient_step_ini = 1
    gradient_step_default = 0.0005
    threshold = 0.000001
    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []
    time_start = process_time()
    x = x0

    ##### Boucle sur les iterations
    for k in range(iter_max):

        # Valeur du critere et du gradient
        ind = 4
        critere, gradient = Oracle(x, ind)

        # Test de convergence
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        D = -gradient

        # Mise a jour des variables
        gradient_step, ok = Wolfe(gradient_step_ini, x, D, Oracle)
        if ok == 2:
            gradient_step = gradient_step_default
        x = x + (gradient_step*D)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

    ##### Resultats de l'optimisation
    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start
    print()
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', np.linalg.norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt

def BFGS(Oracle, x0):

    ##### Initialisation des variables
    iter_max = 10000
    gradient_step = 1
    threshold = 0.000001
    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []
    time_start = process_time()
    x = x0
    N = int(np.shape(x)[0])
    print(N)
    W = np.eye(N,N)
    I = np.eye(N,N)

    ##### Boucle sur les iterations
    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient = Oracle(x)

        # Test de convergence
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        D = -np.dot(W,gradient)

        # Mise a jour des variables
        if k == 0:
            gradient_step, ok = 0.005, 1 #Wolfe(gradient_step_ini, x, D, Oracle)
            x_ant = copy.copy(x)
            x = x + (gradient_step*D)
            critere_ant = critere
        else:
            gradient_step, ok = 0.005, 1 #Wolfe(gradient_step_ini, x, D, Oracle)
            x_ant = copy.copy(x)
            x = x + (gradient_step*D)
            delta_x = x - x_ant
            delta_g = critere - critere_ant
            prod1 = (I - np.dot(delta_x,np.transpose(delta_g))/np.dot(np.transpose(delta_g),delta_x))
            prod2 = (I - np.dot(delta_g,np.transpose(delta_x))/np.dot(np.transpose(delta_g),delta_x))
            prod3 = (I - np.dot(delta_x,np.transpose(delta_x))/np.dot(np.transpose(delta_g),delta_x))
            W = np.dot(np.dot(prod1,W),prod2) + prod3
            critere_ant = critere

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

    ##### Resultats de l'optimisation
    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start
    print()
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', np.linalg.norm(gradient_opt))

    # Visualisation de la convergence
    Visualg(gradient_norm_list, gradient_step_list, critere_list)

    return critere_opt, gradient_opt, x_opt
