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

        self.symb2int = {self.pad_symbol: 2, self.start_symbol:0, self.stop_symbol:1}
        counter_symb = 3

        raw_data = dict()
        with open(filename, 'rb') as fp:
            raw_data = pickle.load(fp)

        # Extraction des symboles
        labels = dict()
        coords = dict()
        for i in range(len(raw_data)):
            labels[i] = raw_data[i][0]
            coords[i] = torch.stack((torch.tensor(raw_data[i][1][0]), torch.tensor(raw_data[i][1][1])))

        # Dictionnaire de symboles
        for i in range(len(labels)):
            label = list(labels[i])
            for symb in label:
                if symb not in self.symb2int:
                    self.symb2int[symb] = counter_symb
                    counter_symb += 1
            labels[i] = label
        self.int2symb = {v:k for k,v in self.symb2int.items()}

        # Ajout du padding aux séquences
        self.max = 0
        for word in labels.values():
            if len(word) >= self.max :
                self.max = len(word)
        for key in labels.keys() :
            labels[key] = labels[key] + [self.stop_symbol]
            if len(labels[key])-1 < self.max:
                num_pads = ((self.max - len(labels[key]) + 1))
                labels[key] = labels[key] + [self.pad_symbol for _ in range(num_pads)]

        self.data = dict()
        for i in range(len(labels)):
            self.data[i] = (labels[i], coords[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][0]
        print(label)
        symb_label = [self.symb2int[i] for i in label]
        return symb_label, self.data[idx][1]

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
