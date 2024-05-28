#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser_result():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--low_out', type=float, default=0.00, help="low outlier")
    parser.add_argument('--high_out', type=float, default=0.00, help="high outlier")
    parser.add_argument('--exp_type', type=str, default='parallel', help='Experiment type')
	
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    args = parser.parse_args()
    return args

	

