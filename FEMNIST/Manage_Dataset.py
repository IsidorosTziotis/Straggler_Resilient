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
import os
import json
from tqdm import tqdm

n_nodes, active_nodes = 20, 1
n_trainset, n_testset = 60000, 10000
b_size_SGD, b_size_averg_SGD, b_size_FedGate = 100, 100, 100

"""
#Merging all the json files into one file for training and one for testing
#Training set
dirname = './train'
all_data_train = {
    'users': [],
    'num_samples': [],
    'user_data': {}
}
for filename in tqdm(os.listdir(dirname)):
    fullpath = os.path.join(dirname, filename)
    trainset = json.load(open(fullpath, 'r'))
    all_data_train['users'].extend(trainset['users'])
    all_data_train['num_samples'].extend(trainset['num_samples'])
    for k, v in trainset['user_data'].items():
        if k in all_data_train['user_data']:
            raise ValueError('Duplicate Users.')
        else:
            all_data_train['user_data'][k] = v

#Test set
dirname = './test'
all_data_test = {
    'users': [],
    'num_samples': [],
    'user_data': {}
}
for filename in tqdm(os.listdir(dirname)):
    fullpath = os.path.join(dirname, filename)
    testset = json.load(open(fullpath, 'r'))
    all_data_test['users'].extend(testset['users'])
    all_data_test['num_samples'].extend(testset['num_samples'])
    for k, v in testset['user_data'].items():
        if k in all_data_test['user_data']:
            raise ValueError('Duplicate Users.')
        else:
            all_data_test['user_data'][k] = v

trainset = all_data_train
testset = all_data_test



#Turning json files to hash tables with all data for train and test
#Training Set
firstsumtest = sum(testset['num_samples'])
firstsumtrain = sum(trainset['num_samples'])
hash_train_x = defaultdict(list)
hash_train_y = defaultdict(list)
hash_test_x = defaultdict(list)
hash_test_y = defaultdict(list)

for key in trainset.keys():
    if type(trainset[key]) == dict:
        newdict = trainset[key]
        for j in newdict.keys():
            myset = set(newdict[j]['y'])
            myset = sorted(myset)
            hash_train_x[tuple(myset)].extend(newdict[j]['x'])
            hash_train_y[tuple(myset)].extend(newdict[j]['y'])
            hash_test_x[tuple(myset)].extend(testset['user_data'][j]['x'])
            hash_test_y[tuple(myset)].extend(testset['user_data'][j]['y'])
            #print("For the combination of labels ", myset, "we get ", type(hash_table_x[tuple(myset)]), len(hash_table_x[tuple(myset)]), len(hash_table_x[tuple(myset)][0]))
    else:
        #print(trainset[key])
        #input("Stop2")
        pass
secondsumtrain = 0
secondsumtest = 0
for key in hash_train_x.keys():
    secondsumtrain += len(hash_train_x[key])
    secondsumtest += len(hash_test_x[key])
    #print("For combination ", key, "we get ", len(hash_train_x[key]), "samples and ", len(hash_train_y[key]), "lables ...and for the test set we get ", len(hash_test_x[key]), "samples and ", len(hash_test_y[key]))
#print ("The first sum for training set is", firstsumtrain, "and the second sample count is ", secondsumtrain)
#print ("The first sum for test set is", firstsumtest, "and the second sample count is ", secondsumtest)


#Turning hash tables to lists for train and test
x_train = []
y_train = []
x_test = []
y_test = []
for key in hash_train_x.keys():
    x_train.extend(hash_train_x[key])
    y_train.extend(hash_train_y[key])
    x_test.extend(hash_test_x[key])
    y_test.extend(hash_test_y[key])
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
print(len(x_train[5]), len(x_test[115]), type(y_train[3]), type(y_test[45]))
ctrain = list(zip(x_train, y_train))
#random.shuffle(ctrain)
#ctrain =ctrain[:15000]
ctest = list(zip(x_test, y_test))
print(len(ctrain), len(ctest))
pickle.dump(ctrain, open("WTrL.dat", "wb"))
pickle.dump(ctest, open("WTtL.dat", "wb"))


#ctrain = ctrain[:5000]
#ctest = ctest[:5000]
#pickle.dump(ctrain, open("WTrL_restricted.dat", "wb"))
#pickle.dump(ctest, open("WTtL_restricted.dat", "wb"))


#Splitting lists to `digits', `upper' and `lower' and subsampling from each category
#Training set
digitsx = []
digitsy = []
upperx = []
uppery = []
lowerx = []
lowery = []
countd, countu, countl = 0, 0, 0

ctrain = pickle.load(open("WTrL.dat", "rb"))
print("Half way there")
for i in range(len(ctrain)):
    if ctrain[i][1] < 10:
        digitsx.append(ctrain[i][0])
        digitsy.append(ctrain[i][1])
        countd += 1
    elif ctrain[i][1] < 36:
        upperx.append(ctrain[i][0])
        uppery.append(ctrain[i][1])
        countu += 1
    else:
        lowerx.append(ctrain[i][0])
        lowery.append(ctrain[i][1])
        countl += 1
print(countd, len(digitsx), len(digitsy), countu, len(upperx), len(uppery),  countl, len(lowerx), len(lowery), sum([countd, countu, countl]))
digits = list(zip(digitsx, digitsy))
random.shuffle(digits)
digits = digits[:20000]
upper = list(zip(upperx, uppery))
random.shuffle(upper)
upper = upper[:40000]
digits.extend(upper)
random.shuffle(digits)
print(type(digits), len(digits))
pickle.dump(digits, open("Train_Digits_Upper_60k.dat", "wb"))

#Verifing numbers
train = pickle.load(open("Train_Digits_Upper_60k.dat", "rb"))
countd, countu, countl = 0, 0, 0
for i in range(len(train)):
    if train[i][1] < 10:
        countd += 1
    elif  train[i][1] < 36:
        countu += 1
    else:
        countl += 1
print(countd, countu, countl)

#Test Set
digitsx = []
digitsy = []
upperx = []
uppery = []
lowerx = []
lowery = []
countd, countu, countl = 0, 0, 0

ctest = pickle.load(open("WTtL.dat", "rb"))
print("Half way there")
for i in range(len(ctest)):
    if ctest[i][1] < 10:
        digitsx.append(ctest[i][0])
        digitsy.append(ctest[i][1])
        countd += 1
    elif ctest[i][1] < 36:
        upperx.append(ctest[i][0])
        uppery.append(ctest[i][1])
        countu += 1
    else:
        lowerx.append(ctest[i][0])
        lowery.append(ctest[i][1])
        countl += 1
print(countd, len(digitsx), len(digitsy), countu, len(upperx), len(uppery),  countl, len(lowerx), len(lowery), sum([countd, countu, countl]))
digits = list(zip(digitsx, digitsy))
random.shuffle(digits)
digits = digits[:3500]
upper = list(zip(upperx, uppery))
random.shuffle(upper)
upper = upper[:6500]
digits.extend(upper)
random.shuffle(digits)
print(type(digits), len(digits))
pickle.dump(digits, open("Test_Digits_Upper_60k.dat", "wb"))

#Verifing
test = pickle.load(open("Test_Digits_Upper_60k.dat", "rb"))
countd, countu, countl = 0, 0, 0
for i in range(len(test)):
    if test[i][1] < 10:
        countd += 1
    elif test[i][1] < 36:
        countu += 1
    else:
        countl += 1
print(countd, countu, countl)
"""


ctrain = pickle.load(open("Train_Digits_Upper_60k.dat", "rb"))
ctest = pickle.load(open("Test_Digits_Upper_60k.dat", "rb"))

input("Hey1")
print(type(ctrain), type(ctest))
print(len(ctrain), len(ctest))
train_class_samples = [0] * 36
test_class_samples = [0]*36
for i,j in ctrain:
    train_class_samples[j] += 1
for i,j in ctest:
    test_class_samples[j] += 1
print(train_class_samples)
print(test_class_samples)
sumtrainlist = []
sumtestlist = []
sumtrain = sumtest = 0
for i in range(36):
    if i == 0:
        sumtrain = train_class_samples[i]
        sumtest = test_class_samples[i]
    else:
        sumtrain = train_class_samples[i] + sumtrainlist[i-1]
        sumtest = test_class_samples[i] + sumtestlist[i-1]
    sumtrainlist.append(sumtrain)
    sumtestlist.append(sumtest)
print(sumtrainlist)
print(sumtestlist)
input("Hey2")
counter = 0
batched_train_x = []
batched_train_y = []
newlist_x = []
newlist_y = []
for i,j in ctrain:
    counter += 1
    newlist_x.append(torch.FloatTensor(i))
    newlist_y.append(j)
    if counter%b_size_FedGate == 0:
        batched_train_x.append(torch.stack(newlist_x))
        batched_train_y.append(torch.FloatTensor(newlist_y))
        newlist_x = []
        newlist_y = []
batched_train = list(zip(batched_train_x, batched_train_y))

node_trainloader = []
counter = 0
#print(len(batched_train))
batches_per_node = len(batched_train)//n_nodes
newlist_x = []
newlist_y = []
for i, j in batched_train:
    counter += 1
    newlist_x.append(i)
    newlist_y.append(j)
    if counter%batches_per_node == 0:
        node_trainloader.append(list(zip(newlist_x, newlist_y)))
        newlist_x = []
        newlist_y = []

counter = 0
batched_train_x = []
batched_train_y = []
newlist_x = []
newlist_y = []
for i,j in ctrain:
    counter += 1
    newlist_x.append(torch.FloatTensor(i))
    newlist_y.append(j)
    if counter%n_trainset == 0:
        batched_train_x.append(torch.stack(newlist_x))
        batched_train_y.append(torch.FloatTensor(newlist_y))
        newlist_x = []
        newlist_y = []
trainloader_whole = list(zip(batched_train_x, batched_train_y))

counter = 0
batched_test_x = []
batched_test_y = []
newlist_x = []
newlist_y = []
for i,j in ctest:
    counter += 1
    newlist_x.append(torch.FloatTensor(i))
    newlist_y.append(j)
    if counter%n_testset == 0:
        batched_test_x.append(torch.stack(newlist_x))
        batched_test_y.append(torch.FloatTensor(newlist_y))
        newlist_x = []
        newlist_y = []
testloader = list(zip(batched_test_x, batched_test_y))

Dataset = [node_trainloader, trainloader_whole, testloader]
pickle.dump(Dataset, open("Femnist_Digits_Upper_60k_Dataset.dat", "wb"))
