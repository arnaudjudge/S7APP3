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

    return mat[l_a-1, l_b-1]

def confusion_matrix(true, pred, ignore=[]):

    # Calcul de la matrice de confusion
    classes = np.unique(true)
    classes = [x for x in classes if x not in ignore]
    conf_mat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for l in range(len(true)):
        t = true[l]
        p = pred[l]
        for i in range(len(classes)):
            for j in range(len(classes)):
                for k in range(len(t)):
                    if (p[k] == classes[i]) & (t[k] == classes[j]):
                        conf_mat[i, j] += 1
    return conf_mat, classes


