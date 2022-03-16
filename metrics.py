# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition

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

    # À compléter

    return None
