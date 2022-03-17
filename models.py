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
        self.hidden_q = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(in_features=2*self.hidden_dim, out_features=self.dict_size)
        self.to(device)

    def encoder(self, x):
        x, h = self.encoder_rnn(x)
        return x, h

    def att_module(self, encoder_out, q):

        q = self.hidden_q(q)
        att_weights = nn.functional.softmax(torch.bmm(encoder_out, q.view(-1, self.hidden_dim, 1)), dim=1)

        att_out = torch.bmm(att_weights.view(-1, 1, self.maxlen['coords']), encoder_out)

        return att_out, att_weights

    def decoder(self, encoder_out, h):
        max_len = self.maxlen['labels']
        batch_size = h.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device) # Vecteur de sortie du décodage
        att_weights = torch.zeros((batch_size, self.maxlen['coords'], self.maxlen['labels'])).to(self.device)

        for i in range(max_len):
            x = self.embedding(vec_in)
            x, h = self.decoder_rnn(x.view(-1, 1, self.hidden_dim), h)

            att_out, w = self.att_module(encoder_out=encoder_out, q=x)
            x = torch.cat((x.reshape(-1, self.hidden_dim), att_out.reshape(-1, self.hidden_dim)), dim=1)
            x = self.fc(x)

            vec_out[:, i, :] = x
            att_weights[:, :, i] = w.view(-1, self.maxlen['coords'])
            vec_in = torch.argmax(x, dim=1).to(self.device).long()

        return vec_out, h, att_weights

    def forward(self, x, h=None):

        x, h = self.encoder(x)
        x, h, w = self.decoder(x, h)

        return x, h, w
    

