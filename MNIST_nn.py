from copy import deepcopy
import torch
from torch import nn
from torchvision import datasets, transforms
from MNIST_nn_methods import nn_SGD, nn_averg_SGD, FedGate, Plotter, FedNova
import math
import random
import pickle

n_nodes, active_nodes = 20, 1
n_trainset, n_testset = 60000, 10000
b_size_SGD, b_size_averg_SGD, b_size_FedGate = 100, 100, 100
eta_SGD, eta_averg_SGD, eta_FedGate = 0.005, 0.005, 0.005
epochs_SGD, epochs_averg_SGD, epochs_FedGate = 15, 62, 62
threshold_FedGate, threshold_factor = 100, 2
gamma, doubling_factor = 1, 1.3
input_size, output_size = 784 ,10
H1, H2 = 128, 64
steps = math.floor(math.log(n_nodes, 2))
#steps = 0
lower_speed, top_speed = 50, 500


#Creating speeds uniformly at random from lower_speed to top_speed
List_of_speeds = []
for i in range(n_nodes):
    List_of_speeds.append(random.randint(lower_speed, top_speed))
List_of_speeds.sort()
List_of_speeds[0]= 50
List_of_speeds[n_nodes-1] = 500

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
currentset = trainset
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

#Split the data for each node i, to train_set[i] and creating corresponding trainloaders
train_set = []
node_trainloader = []
node_trainldr = []
for i in range(n_nodes):
    temp_set, currentset = torch.utils.data.random_split(currentset, [int(n_trainset/n_nodes), int(len(currentset)-n_trainset/n_nodes)])
    train_set.append(temp_set)
    ndtrainldr = torch.utils.data.DataLoader(train_set[i], batch_size=b_size_FedGate, shuffle=True)
    ndtrainldr2 = torch.utils.data.DataLoader(train_set[i], batch_size=int(n_trainset/active_nodes), shuffle=True)
    node_trainloader.append(ndtrainldr)
    node_trainldr.append(ndtrainldr2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size_FedGate, shuffle=True)
trainloader_whole = torch.utils.data.DataLoader(trainset, batch_size=n_trainset, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=n_testset, shuffle=True)

#Defining the model
model = nn.Sequential(nn.Linear(input_size, H1),
                      nn.ReLU(),
                      nn.Linear(H1, H2),
                      nn.ReLU(),
                      nn.Linear(H2, output_size),
                      nn.LogSoftmax(dim=1))
model_SGD = deepcopy(model)
model_averg_SGD = deepcopy(model)
model_FedNova = deepcopy(model)
model_FedGate_doubling = deepcopy(model)
model_FedGate_allnodes = deepcopy(model)

#Choose methods to run
#nn_SGD(model_SGD, trainloader_whole, trainloader, testloader, epochs_SGD, eta_SGD, n_trainset, n_testset, b_size_SGD)
#input("Stop after returning")
returning_values_FedAvg = nn_averg_SGD(model_averg_SGD, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_averg_SGD, eta_averg_SGD, n_trainset, n_testset, b_size_averg_SGD, n_nodes, List_of_speeds)
pickle.dump(returning_values_FedAvg, open("MNIST_nn_FedAvg_last.dat", "wb"))
#input("Stop after returning")
returning_values_FedNova = FedNova(model_FedNova, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_averg_SGD, eta_averg_SGD, n_trainset, n_testset, b_size_averg_SGD, n_nodes, List_of_speeds)
pickle.dump(returning_values_FedNova, open("MNIST_nn_FedNova_last.dat", "wb"))
#input("Stop after returning")
returning_values_FedGate_Doubling = FedGate(model_FedGate_doubling, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_FedGate, eta_FedGate *((doubling_factor)**steps), n_trainset, n_testset, b_size_averg_SGD, n_nodes, active_nodes, gamma / ((doubling_factor)**steps), threshold_FedGate, doubling_factor, threshold_factor, List_of_speeds)
pickle.dump(returning_values_FedGate_Doubling, open("MNIST_nn_Doubling_last.dat", "wb"))
#input("Stop here")
returning_values_FedGate_Allnodes = FedGate(model_FedGate_allnodes, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_FedGate, eta_FedGate, n_trainset, n_testset, b_size_averg_SGD, n_nodes, n_nodes, gamma, threshold_FedGate, doubling_factor, threshold_factor, List_of_speeds)
pickle.dump(returning_values_FedGate_Allnodes, open("MNIST_nn_Allnodes_last.dat", "wb"))
#input("Stop here")
Plotter()
