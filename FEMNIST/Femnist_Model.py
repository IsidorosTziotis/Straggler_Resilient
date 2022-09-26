from torch import nn
from copy import deepcopy
import pickle

input_size, output_size = 784 ,36
H1, H2 = 128, 64

#Initializing a model
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
model_FLANP_FedAvg = deepcopy(model)
model_FLANP_FedNova = deepcopy(model)

pickle.dump(model_averg_SGD, open("Model_FedAvg.dat", "wb"))
pickle.dump(model_FedNova, open("Model_FedNova.dat", "wb"))
pickle.dump(model_FedGate_allnodes, open("Model_FedGATE.dat", "wb"))
pickle.dump(model_FedGate_doubling, open("FLANP.dat", "wb"))