from __future__ import division
import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
from collections import defaultdict
from PIL import Image
import torch
from torch import nn
from copy import deepcopy
import pickle
from Methods_Femnist import  nn_averg_SGD, FedNova, FedGate, Plotter

n_nodes, active_nodes = 20, 1
n_trainset, n_testset = 60000, 10000
b_size_SGD, b_size_averg_SGD, b_size_FedGate = 100, 100, 100
eta_averg_SGD, eta_FedGate = 0.012, 0.012
epochs_averg_SGD, epochs_FedGate = 400, 400
threshold_FedGate, threshold_factor = 10000, 2
gamma, doubling_factor = 1, 1.4
steps = math.floor(math.log(n_nodes, 2))


#Loading Dataset
Dataset = pickle.load(open("Femnist_Digits_Upper_60k_Dataset.dat", "rb"))
node_trainloader = Dataset[0]
trainloader_whole = Dataset[1]
testloader = Dataset[2]
#Loading Speeds
List_of_speeds = pickle.load(open("Exp_Speeds.dat", "rb"))
#model_choice = input("For a fresh model press 0, for continuing press any other key")
model_choice = '0'
#Loading models
if model_choice == '0':
    model_averg_SGD = pickle.load(open("Model_FedAvg.dat", "rb"))
    model_FedNova = pickle.load(open("Model_FedNova.dat", "rb"))
    model_FedGate_allnodes = pickle.load(open("Model_FedGATE.dat", "rb"))
    model_FedGate_doubling = pickle.load(open("FLANP.dat", "rb"))
else:
    model_averg_SGD = pickle.load(open("LR_Model_FedAvg.dat", "rb"))
    model_FedNova = pickle.load(open("LR_Model_FedNova.dat", "rb"))
    model_FedGate_allnodes = pickle.load(open("LR_Model_FedGATE.dat", "rb"))
    model_FedGate_doubling = pickle.load(open("LR_FLANP.dat", "rb"))

node_trainldr = 0

returning_values_FedAvg = nn_averg_SGD(model_averg_SGD, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_averg_SGD, eta_averg_SGD, n_trainset, n_testset, b_size_averg_SGD, n_nodes, List_of_speeds)
pickle.dump(returning_values_FedAvg, open("Femnist_FedAvg.dat", "wb"))

returning_values_FedNova = FedNova(model_FedNova, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_averg_SGD, eta_averg_SGD, n_trainset, n_testset, b_size_averg_SGD, n_nodes, List_of_speeds)
pickle.dump(returning_values_FedNova, open("Femnist_FedNova.dat", "wb"))

returning_values_FedGate_Allnodes = FedGate(model_FedGate_allnodes, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_FedGate, eta_FedGate, n_trainset, n_testset, b_size_averg_SGD, n_nodes, n_nodes, gamma, threshold_FedGate, doubling_factor, threshold_factor, List_of_speeds)
pickle.dump(returning_values_FedGate_Allnodes, open("Femnist_FedGATE.dat", "wb"))

returning_values_FedGate_Doubling = FedGate(model_FedGate_doubling, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_FedGate, eta_FedGate *((doubling_factor)**steps), n_trainset, n_testset, b_size_averg_SGD, n_nodes, active_nodes, gamma / ((doubling_factor)**steps), threshold_FedGate, doubling_factor, threshold_factor, List_of_speeds)
pickle.dump(returning_values_FedGate_Doubling, open("Femnist_FLANP.dat", "wb"))

