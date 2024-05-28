from numpy import * 
import pickle as pkl
import math
import pandas as pd

import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch, os, gc
import json
import random
import argparse
from util2.options import args_parser
from models2.Update import LocalUpdateFed
from models2.Nets import AirModel
from models2.Fed import FA
from models2.test import test_imgCE
from torch.utils.data import DataLoader, Dataset


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') 
    print(args)
    
    ###dataset preprocessing and splitting####
    df = pd.read_csv('Dataset.csv')
    df.columns
    timeseries = df["People"].values.astype('float32')
    train_size = int(len(timeseries) * 0.80)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    ###Local training datasets generation for federated client####
    X_train=train[0:train_size]
    no_of_users=20
    num_items = int(train_size/no_of_users)
    dict_users, all_idxs = {}, [i for i in range(train_size)]
    for i in range(no_of_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])  
    local_datasets_sample=[]
    for user in range (no_of_users):
        index_list=list(dict_users[user])
        local_datasets_sample.append(train[index_list])
    local_datasets_sample_2=[]
    local_datasets_label=[]
    train_window = 5    
    for i in range (no_of_users):
        train_inout_seq = create_inout_sequences(torch.from_numpy(local_datasets_sample[i]), train_window)
        local_datasets_sample_2.append(train_inout_seq)
    
    ###Global testing dataset for testing the global model####    
    test_inout_seq = create_inout_sequences(torch.from_numpy(test), train_window)
    
    
    ###Global model initialization####    
    net_glob_FACE = AirModel().to(args.device)         

    ###Folder creation for storing the result file####      
    folder="Result_Folder"
    if not os.path.isdir(folder): os.makedirs(folder)  
    
    
    ####
    training_loss_avg = dict.fromkeys((range(args.epochs)),0)
    prediction_loss_avg=dict.fromkeys((range(args.epochs)),0)
    MyDicts=[]
    
    
    ###Global training rounds for federated learning###
    for iter in range(args.epochs): 
        print ("......................Global Epoch......................:", iter)
        
        ####Client selection for training####
        m = max(int(args.frac * args.num_users), 1)        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)   
        w_locals_FACE, loss_locals_FACE= [], []    
        
        ####Training models on the selected clients#####
        for idx in idxs_users:
            client=idx
            local_Fed = LocalUpdateFed(args=args, train_inout_seq=local_datasets_sample_2[idx], idxs=dict_users[idx], iter=iter+idx+1) 
            #Training
            print ('..........client............', idx)            
            net=copy.deepcopy(net_glob_FACE).to(args.device)
            w_FACE, loss_FACE = local_Fed.trainFACE(net=net)
            w_locals_FACE.append(copy.deepcopy(w_FACE))
            loss_locals_FACE.append(loss_FACE)
            
        ### Global model update using local models' weight aggregation/averaging####
        w_glob_FACE = FA(w_locals_FACE, args)
        # copy weight to net_glob
        net_glob_FACE.load_state_dict(w_glob_FACE)
        
        # Update average training loss
        loss_avg_FACE = sum(loss_locals_FACE) / len(loss_locals_FACE) 
        training_loss_avg.update({iter:loss_avg_FACE}) 
        
        #Testing global model  
        net_glob_FACE.eval() 
        prediction_loss_avg[iter]= test_imgCE(net_glob_FACE,test_inout_seq)               
     
    ### Storing training_loss_avg and prediction_loss_avg into MyDicts so that this torch file can be reopened later for result analysis and plotting purpoes####
    MyDicts = [training_loss_avg,prediction_loss_avg]
    f1="result_file"
    filename_mydicts = os.path.join(folder,f1) 
    torch.save(MyDicts, filename_mydicts)  
    