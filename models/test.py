from sklearn.metrics import *
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import numpy as np
from torch.autograd import Variable
import gc
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import numpy as np
import torch.utils.data as data



def test_imgCE(net_g, test):
    net_g.eval()
    all_loss = []
    with torch.no_grad():
        for seq, labels in test:
            net_g.hidden = (torch.zeros(1, 1, net_g.hidden_layer_size),
                            torch.zeros(1, 1, net_g.hidden_layer_size))
            y_pred = net_g(seq)
            single_loss =metrics.mean_squared_error(y_pred, labels)
            all_loss.append(single_loss)
        prediction_loss=sum(all_loss) / (len(all_loss)+0.001)         
        print("prediction_loss", prediction_loss)
        
    return prediction_loss

