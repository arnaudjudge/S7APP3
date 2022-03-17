# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

import dataset
from models import *
from dataset import *
from metrics import *
if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    batch_size = 80
    n_hidden = 25
    n_layers = 3
    # À compléter
    n_epochs = 20

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # À compléter

    dataset_words = HandwrittenWords("data_trainval.p")
    
    # Séparation de l'ensemble de données (entraînement et validation)
    # À compléter
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_words,
                                                               [int(len(dataset_words) * 0.7),
                                                                int(len(dataset_words) - int(
                                                                    len(dataset_words) * 0.7))])

    # Instanciation des dataloaders
    # À compléter
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    # Instanciation du model
    # À compléter
    model = trajectory2seq(hidden_dim=n_hidden, n_layers=n_layers,
                    symb2int=dataset.symb2int, int2symb=dataset.int2symb,
                    dict_size=dataset.dict_size, device=device, max_len=dataset.max_len)

    # Initialisation des variablesDtaset initial
    # À compléter

    if trainning:

        # Fonction de coût et optimizateur
        # À compléter

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            # À compléter
            
            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass
