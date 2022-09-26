import random
from tqdm import trange
from collections import defaultdict
from PIL import Image
import torch
from torch import nn
from copy import deepcopy
import pickle
import numpy as np

n_nodes, active_nodes = 20, 1
lower_speed, top_speed = 50, 500

#Exponential Speeds
List_of_speeds = []
for i in range(n_nodes):
    List_of_speeds.append(np.random.exponential(225)+lower_speed)
List_of_speeds.sort()
#print(List_of_speeds)

pickle.dump(List_of_speeds, open("Exp_Speeds.dat", "wb"))