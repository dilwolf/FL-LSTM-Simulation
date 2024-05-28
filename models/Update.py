
import copy
import math
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import pandas as pd
from torch.autograd import Variable
import torch.nn.init as init
from collections import OrderedDict
import gc
from sklearn import preprocessing
import torch.optim as optim
import numpy as np
import torch.utils.data as data


class LocalUpdateFed(object):
    def __init__(self, args, train_inout_seq, idxs=None, iter=iter):  
        self.args = args
        self.loss_fn = nn.MSELoss()
        self.selected_clients = []
        self.epoch_client=iter
        self.train_inout_seq=train_inout_seq
        self.idxs=idxs



    def trainFACE(self, net):
        net.train()        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr)

        for iter in range(self.args.local_ep):
            print("Local Epoch: ", iter)          
            all_loss=[]
            for seq, labels in self.train_inout_seq:
#                print("(seq)",(seq))
#                print("(labels)",(labels))
                optimizer.zero_grad()
                net.hidden_cell = (torch.zeros(1, 1, net.hidden_layer_size),
                                torch.zeros(1, 1, net.hidden_layer_size))
                y_pred = net(seq)
                single_loss = self.loss_fn(y_pred, labels)
                all_loss.append(single_loss.item())
                single_loss.backward()
                optimizer.step()    
            training_loss=sum(all_loss) / (len(all_loss)+0.001)         
            print("training_loss",training_loss)

        return net.state_dict(), training_loss   
