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

        self.symb_dict = {self.pad_symbol: 2, self.start_symbol:0, self.stop_symbol:1}
        counter_symb = 3

        self.raw_data = dict()
        with open(filename, 'rb') as fp:
            self.raw_data = pickle.load(fp)

        # Extraction des symboles
        self.data = dict()
        for i in range(len(self.raw_data)):
            #creer le dict avec tout les symboles
            label = list(self.raw_data[i][0])
            for symb in label:
                if symb not in self.symb_dict:
                    self.symb_dict[symb] = counter_symb
                    counter_symb += 1
            coords = torch.stack((torch.tensor(self.raw_data[i][1][0]), torch.tensor(self.raw_data[i][1][1])))
            # coords = (self.raw_data[i][1][0], self.raw_data[i][1][1])
            self.data[i] = (label, coords)

        # Ajout du padding aux séquences
        self.max_len = dict()
        self.max_len['labels'] = len(max(self.data[0], key=len)) + 1
        print(self.max_len['labels'])
        print(self.data[0])
        self.max_len['coords'] = max(self.data[1][0].shape, key=np.size) + 1
        print(self.max_len['coords'])
        for i in range(len(self.data[0])):
            self.data[0] = (self.data[0][i] + [self.stop_symbol] + [self.pad_symbol for _ in range(self.max_len['labels'] - len(self.data[0][i]) - 1)], self.data[1][i])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][0]
        symb_label = [self.symb_dict[i] for i in label]
        return torch.tensor(symb_label), self.data[idx][1]

    def visualisation(self, idx):
        # Visualisation des échantillons
        item = self.__getitem__(idx)
        plt.figure()
        plt.plot(item[1][0], item[1][1], '-bo')
        plt.title(item[0])
        plt.show()
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
