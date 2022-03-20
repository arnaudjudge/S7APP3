# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import numpy as np
import torch


def edit_distance(x,y):
    # Calcul de la distance d'édition
##Methode matricielle

    a = ['#'] + x # padding
    b = ['#'] + y
    l_a = len(a)
    l_b = len(b)

    mat = np.zeros((l_a, l_b))
    for i in range(l_a):
        mat[i, 0] = i
    for j in range(l_b):
        mat[0, j] = j

    for i in range(1, l_a):
        for j in range(1, l_b):
            if a[i] == b[j]:
                cost = 0
            else:
                cost = 1

            mat[i, j] = min(mat[i-1, j]+1,
                            mat[i, j-1]+1,
                            mat[i-1, j-1]+cost)
        # Méthode recursive
        # if len(x) == 0:
        #     distance = len(y)
        # elif len(y) == 0:
        #     distance = len(x)
        # elif x[0] == y[0]:
        #     distance = edit_distance(x[1:], y[1:])
        # else:
        #     distance = 1 + min(edit_distance(x[1:], y), edit_distance(x, y[1:]), edit_distance(x[1:], y[1:]))

    return mat[l_a-1, l_b-1]

def confusion_matrix(true, pred, ignore=[]):

    # Calcul de la matrice de confusion
    conf_mat = torch.zeros(26, 26)
    alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
    for i in alph:
        for j in alph:
            conf_mat[i][j] = ((pred == i) & (true == j)).sum.item()
    return conf_mat
    # À compléter

