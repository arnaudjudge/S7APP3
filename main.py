# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import matplotlib.pyplot as plt
import matplotlib.colors
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    seed = 0                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    reuse_model = False
    train_val_split = 0.8
    test_length = 5
    batch_size = 64
    n_epochs = 100
    lr = 0.01

    n_hidden = 15
    n_layers = 2

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    ds = HandwrittenWords('data_trainval.p')

    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(ds)*train_val_split)
    n_val_samp = len(ds)-n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(ds, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    if reuse_model:
        model = torch.load('model_BI_ATTN.pt', map_location=lambda storage, loc: storage)
        model = model.to(device)
        print('model loaded from file')
    else:
        model = trajectory2seq(hidden_dim=n_hidden, n_layers=n_layers, int2symb=ds.int2symb, symb2int=ds.symb2int, \
                               dict_size=len(ds.int2symb), device=device, maxlen=ds.max_len)

    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))

    # Initialisation des variables
    best_val_loss = np.inf

    if trainning:
        fig, ax = plt.subplots(1, 2) # Initialisation figure
        train_loss = []
        val_loss = []
        edit_dist_train = []
        edit_dist_val = []

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2) # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}")
            # Entraînement

            running_loss_train = 0
            dist_t = 0
            model.train()
            for batch_idx, data in enumerate(dataload_train):
                labels, writing = data
                writing = writing.to(device).float()
                labels = labels.to(device).long()

                pred, hidden, att_weights = model(writing)

                loss = criterion(pred.view((-1, model.dict_size)), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss_train += loss.item()
                dist_t = 0
                pred_word = torch.argmax(pred, dim=2)
                for idx in range(len(labels)):
                    p = pred_word[idx].detach().cpu().tolist()
                    l = labels[idx].detach().cpu().tolist()
                    # symb_p = [ds.int2symb[i] for i in p]
                    # symb_l = [ds.int2symb[i] for i in l]
                    M = l.index(1)
                    dist_t += edit_distance(p[:M],l[:M])/len(labels)
            edit_dist_train.append(dist_t)

            # Validation
            running_loss_val = 0
            dist_v = 0
            model.eval()
            for data in dataload_val:
                labels, writing = data
                writing = writing.to(device).float()
                labels = labels.to(device).long()

                pred, hidden, att_weights = model(writing)
                loss = criterion(pred.view((-1, model.dict_size)), labels.view(-1))
                running_loss_val += loss.item()

                pred_word = torch.argmax(pred, dim=2)
                for idx in range(len(labels)):
                    p = pred_word[idx].detach().cpu().tolist()
                    l = labels[idx].detach().cpu().tolist()
                    # symb_p = [ds.int2symb[i] for i in p]
                    # symb_l = [ds.int2symb[i] for i in l]
                    M = l.index(1)
                    dist_v += edit_distance(p[:M],l[:M])/len(labels)
            edit_dist_val.append(dist_v/len(dataload_val))

            # Ajouter les loss aux listes
            train_loss.append(running_loss_train/len(dataload_train))
            val_loss.append(running_loss_val/len(dataload_val))

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                print('saving new best model')
                best_val_loss = running_loss_val
                torch.save(model, 'model.pt')

            # Affichage
            print(f"Training loss: {running_loss_train/len(dataload_train)}, edit distance: {dist_t}")
            print(f"Validation loss: {running_loss_val/len(dataload_val)}, edit distance: {dist_v/len(dataload_val)}")
            print("\n")
            if learning_curves:
                # visualization
                ax[0].cla()
                ax[0].plot(train_loss, label='training loss')
                ax[0].plot(val_loss, label='validation loss')
                ax[0].legend()
                ax[1].cla()
                ax[1].plot(edit_dist_train, '--', label='train edit distance')
                ax[1].plot(edit_dist_val, '--', label='validation edit distance')
                ax[1].legend()
                # plt.draw()
                # plt.pause(0.01)
                plt.savefig('learning_curves.png')
                #plt.show()

    if test:
        # Évaluation
        model = torch.load('model.pt', map_location=lambda storage, loc: storage)
        model = model.to(device)
        model.eval()

        # Charger les données de tests
        # prendre le val
        preds = []
        dist_test = 0
        for data in dataload_val:
            labels, writing = data
            writing = writing.to(device).float()
            labels = labels.to(device).long()

            pred, hidden, att_weights = model(writing)
            pred_word = torch.argmax(pred, dim=2)

            for idx in range(len(labels)):
                p = pred_word[idx].detach().cpu().tolist()
                l = labels[idx].detach().cpu().tolist()
                symb_p = [ds.int2symb[i] for i in p]
                symb_l = [ds.int2symb[i] for i in l]
                M = l.index(1)
                d = edit_distance(p[:M],l[:M])
                print(f"""Label: {symb_l[:symb_l.index('<eos>')+1]}\nPred: {symb_p[:symb_p.index('<eos>')+1]}\n--- Edit distance: {d}\n\n""")
                dist_test += d/len(labels)
        dist_test = dist_test / len(dataload_val)
        print(f'Test edit distance: {dist_test}')


        # Affichage des résultats de test
        for i in range(test_length):
            rand_item = ds[np.random.randint(0, len(ds))]

            w = rand_item[1].to(device).float()
            p, _, attn = model(w.reshape(1, 457, 2))
            p = torch.argmax(p, dim=2).reshape(6).detach().cpu().tolist()
            l = rand_item[0].detach().cpu().tolist()

            symb_p = [ds.int2symb[i] for i in p]
            symb_l = [ds.int2symb[i] for i in l]
            print(symb_l[:symb_l.index('<eos>')+1])
            print(symb_p[:symb_p.index('<eos>')+1])
            M = symb_l.index('<eos>')
            print(edit_distance(symb_p[:M], symb_l[:M]))
            print('\n')

            x, y = w.cpu().detach().numpy().T
            # plt.figure()
            # plt.plot(x, y, '-bo')
            # plt.show()

            # Affichage de l'attention
            attn = attn.cpu().detach().cpu()
            fig, ax = plt.subplots(6, 1)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', ['#d0d0d0', '#000000'], N=100)
            fig.suptitle(symb_l)
            for i in range(6):
                ax[i].scatter(x, y, c=attn[0, :, i], s=1,  cmap=cmap)
                ax[i].set_ylabel(symb_p[i])
            plt.show()

        # Affichage de la matrice de confusion
        # À compléter

