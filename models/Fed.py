
import gc
import copy
import torch
from torch import nn
import math
import numpy as np
import itertools
from itertools import compress  
import statistics
import torch.nn.functional as F
from itertools import product
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.stats.mstats import trim

def FA(w, args):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    del w 
#    print ("w_avg FACE",w_avg)
    if str(args.device)!='cpu':
        with torch.cuda.device(args.device): 
            torch.cuda.empty_cache()       
    gc.collect()
    return w_avg

