# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn

        self.encoder_rnn = nn.GRU(input_size=2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.decoder_rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.embedding = nn.Embedding(num_embeddings=self.dict_size, embedding_dim=self.hidden_dim)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.dict_size)
        self.to(device)

    def encoder(self, x):
        x, h = self.encoder_rnn(x)
        return x, h

    def decoder(self, x, h):
        max_len = self.maxlen['labels']
        batch_size = h.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device) # Vecteur de sortie du décodage

        for i in range(max_len):
            x = self.embedding(vec_in)
            x, h = self.decoder_rnn(x.view(-1, 1, self.hidden_dim), h)

            x = self.fc(x.reshape(-1, self.hidden_dim))

            vec_out[:, i, :] = x
            vec_in = torch.argmax(x, dim=1).to(self.device).long()

        return vec_out, h

    def forward(self, x, h=None):

        x, h = self.encoder(x)
        x, h = self.decoder(x, h)

        return x, h
    

