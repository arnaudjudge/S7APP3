# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.raw_data = dict()
        with open(filename, 'rb') as fp:
            self.raw_data = pickle.load(fp)

        # Extraction des symboles
        self.data = dict()
        for i in range(len(self.raw_data)):
            self.data[i] = (self.raw_data[i][0], torch.stack((torch.tensor(self.raw_data[i][1][0]), torch.tensor(self.raw_data[i][1][1]))))


        print(1)
        # Ajout du padding aux séquences
        chars_dict = {}

        for i in range(len(self.data)):
            chars_dict[i] = [char for char in self.data[i][0]]
        print(chars_dict[57])

        self.max = 0
        for word in chars_dict.values():
            if len(word) >= self.max :
                self.max = len(word)
        for key in chars_dict.keys() :
            chars_dict[key] = chars_dict[key] + [self.stop_symbol]
            if len(chars_dict[key])-1 < self.max:
                self.num_pads = ((self.max - len(chars_dict[key]) + 1))
                for i in range(self.num_pads) :
                    chars_dict[key] = chars_dict[key] + [self.pad_symbol]

        print(chars_dict[785])
        # À compléter
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def visualisation(self, idx):
        # Visualisation des échantillons
        # À compléter (optionel)
        pass
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
