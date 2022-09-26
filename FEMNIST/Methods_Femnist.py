from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import math


def optimality_measures(model, trainloader_whole, n_trainset, criterion, e):
    trainrunning_loss = 0
    trainrunning_misclass_accuracy = 0
    trainrunning_grad = 0
    traincounter = 0
    # Computing training grad and misclassification accuracy
    for trainimages, trainlabels in trainloader_whole:
        # Flatten MNIST images into a 784 long vector
        trainimages = trainimages.view(trainimages.shape[0], -1)
        trainlabels = trainlabels.long()
        """
        print("Images", type(trainimages), type(trainimages[10]))
        print(len(trainimages), trainimages.shape, len(trainimages), len(trainimages[0]))
        print("Labels", type(trainlabels))
        print(len(trainlabels), trainlabels.shape)
        print(trainlabels)
        """
        # Training pass
        # optimizer.zero_grad()
        model.zero_grad()
        trainoutput = model(trainimages)
        # Computing misclassification accuracy
        traintotal = 0
        trainsuccess = 0
        for i in range(n_trainset):
            traintotal += 1
            if trainoutput[i].argmax(0) == trainlabels[i]:
                trainsuccess += 1
        trainmisclass_accuracy = trainsuccess / traintotal

        trainloss = criterion(trainoutput, trainlabels)
        trainloss.backward()
        squared_grad_norm = 0
        cnt = 0
        for p in model.parameters():
            squared_grad_norm += (torch.norm(p.grad)) ** 2
            cnt += 1
        trainrunning_grad += squared_grad_norm
        trainrunning_loss += trainloss.item()
        trainrunning_misclass_accuracy += trainmisclass_accuracy

    return trainrunning_grad, trainrunning_loss / len(trainloader_whole),  trainrunning_misclass_accuracy / len(trainloader_whole)


def nn_averg_SGD(model, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_averg_SGD, eta_averg_SGD, n_trainset, n_testset, b_size_averg_SGD, n_nodes, List_of_speeds):
    print("Running FedAvg")
    TestAccuracy = []
    ActiveGradSquared = []
    TrainGradSquared = []
    TrainLoss = []
    Machine_time = [1]
    # Defining models for each node and an optimizer
    list_of_models = []
    optimizers = []
    for i in range(n_nodes):
        model2 = deepcopy(model)
        #if torch.cuda.is_available():
        #    model2 = model2.cuda()
        list_of_models.append(model2)
        optimizers.append(torch.optim.SGD(list_of_models[i].parameters(), lr=eta_averg_SGD))
    # Define the loss
    criterion = nn.NLLLoss()
    epochs = epochs_averg_SGD
    for e in range(epochs):
        print("FedAvg ", e)
        # Add clock time
        if e == 0:
            Time_FedGate = []
            Time_FedGate.append(List_of_speeds[n_nodes - 1])
        else:
            var = Time_FedGate[e - 1] + List_of_speeds[n_nodes - 1]
            #print("Time added ", List_of_speeds[n_nodes - 1])
            Time_FedGate.append(var)
        for j in range(n_nodes):
            for images, labels in node_trainloader[j]:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
                labels = labels.long()
                """
                print("Images", type(images), type(images[10]))
                print(len(images), images.shape, len(images), len(images[0]) )
                print("Labels", type(labels))
                print(len(labels), labels.shape)
                print(labels)
                print(labels)
                input("check")
                """
                # Training pass
                output = list_of_models[j](images)
                #print("The output is ", type(output), output.shape, output)
                #input("check")
                # Computing Misclassification
                loss = criterion(output, labels)
                optimizers[j].zero_grad()
                list_of_models[j].zero_grad()
                loss.backward()
                # Utilizing the optimizer
                # optimizers[j].step()

                # Manual Optimization Scheme
                with torch.no_grad():
                    for p in list_of_models[j].parameters():
                        new_val = p - eta_averg_SGD * p.grad
                        #optimizers[j].step()
                        p.copy_(new_val)
            #Getting the average model
            if j == 0:
                sdA = list_of_models[j].state_dict()
                for key in sdA:
                    sdA[key] = (sdA[key]/(n_nodes))
                model.load_state_dict(sdA)
            else:
                sdA = model.state_dict()
                sdB = list_of_models[j].state_dict()
                # Average all parameters
                for key in sdB:
                    sdA[key] = sdA[key] + (sdB[key] /(n_nodes))
                model.load_state_dict(sdA)

        #Computing optimality measures w.r.t. the training set
        trainrunning_grad, trainrunning_loss, trainrunning_misclass_accuracy = optimality_measures(model, trainloader_whole, n_trainset, criterion, e)
        print(f"Training loss: ",trainrunning_loss, " and Training misclassification accuracy ", trainrunning_misclass_accuracy )
        #print("Training misclassification accuracy ", trainrunning_misclass_accuracy,"and the gradient squared is", trainrunning_grad)

        #Computing optimality measures w.r.t. the test set
        testrunning_grad, testrunning_loss, testrunning_misclass_accuracy = optimality_measures(model, testloader, n_testset, criterion, e)
        print(f"Test loss: ", testrunning_loss, "Test misclassification accuracy ", testrunning_misclass_accuracy)
        #print("Test misclassification accuracy ", testrunning_misclass_accuracy, "and the test gradient squared is", testrunning_grad)
        TestAccuracy.append(testrunning_misclass_accuracy)
        TrainGradSquared.append(trainrunning_grad)

        #Updating the local models
        sdC = model.state_dict()
        for k in range(n_nodes):
            list_of_models[k].load_state_dict(sdC)

    returning_values = []
    returning_values.append(Machine_time)
    returning_values.append(TrainGradSquared)
    returning_values.append(TestAccuracy)
    returning_values.append(Time_FedGate)
    returning_values.append(TrainLoss)
    returning_values.append(list_of_models[0])
    pickle.dump(list_of_models[0], open("LR_Model_FedAvg.dat", "wb"))
    return returning_values

def FedNova(model, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_averg_SGD, eta_averg_SGD, n_trainset, n_testset, b_size_averg_SGD, n_nodes, List_of_speeds):
    print("Running FedNova")
    TestAccuracy = []
    ActiveGradSquared = []
    TrainGradSquared = []
    TrainLoss = []
    Machine_time = [1]
    #Computing how many epochs each node does per communication. The cap is the time the slowest node needs to complete an epoch
    Taus = []
    sum_reversed_Taus = 0
    for i in range(n_nodes):
        Taus.append(math.floor(List_of_speeds[n_nodes-1]/List_of_speeds[i]))
        sum_reversed_Taus+=1/Taus[i]
    Tau_avg = sum(Taus)/len(Taus)
    sum_reversed_Taus *= Tau_avg
    norm_factor = 1/sum_reversed_Taus
    # Defining models for each node and an optimizer
    list_of_models = []
    optimizers = []
    for i in range(n_nodes):
        model2 = deepcopy(model)
        list_of_models.append(model2)
        optimizers.append(torch.optim.SGD(list_of_models[i].parameters(), lr=eta_averg_SGD))
    # Define the loss
    criterion = nn.NLLLoss()
    epochs = epochs_averg_SGD
    for e in range(epochs):
        print("FedNova ", e)
        # Add clock time
        if e == 0:
            Time_FedGate = []
            Time_FedGate.append(List_of_speeds[n_nodes - 1])
        else:
            var = Time_FedGate[e - 1] + List_of_speeds[n_nodes - 1]
            print("Time added ", List_of_speeds[n_nodes - 1])
            Time_FedGate.append(var)
        for j in range(n_nodes):
            for l in range(Taus[j]):
                for images, labels in node_trainloader[j]:
                    # Flatten MNIST images into a 784 long vector
                    images = images.view(images.shape[0], -1)
                    labels = labels.long()
                    # Training pass
                    output = list_of_models[j](images)
                    # Computing Misclassification
                    loss = criterion(output, labels)
                    optimizers[j].zero_grad()
                    list_of_models[j].zero_grad()
                    loss.backward()
                    # Utilizing the optimizer
                    # optimizers[j].step()

                    # Manual Optimization Scheme
                    with torch.no_grad():
                        for p in list_of_models[j].parameters():
                            new_val = p - eta_averg_SGD * p.grad
                            p.copy_(new_val)
            #Getting the average model
            if j == 0:
                sdA = list_of_models[j].state_dict()
                for key in sdA:
                    #print("Before ", sdA[key])
                    sdA[key] = (sdA[key]/(1))*(Tau_avg/Taus[j])*norm_factor
                    #print("After ", sdA[key])
                    #input("press")
                    #temp = sdA[key].grad
                model.load_state_dict(sdA)
            else:
                sdA = model.state_dict()
                sdB = list_of_models[j].state_dict()
                # Average all parameters
                for key in sdB:
                    sdA[key] = sdA[key] + (sdB[key]/1)*(Tau_avg/Taus[j])*norm_factor
                model.load_state_dict(sdA)

        #Computing optimality measures w.r.t. the training set
        trainrunning_grad, trainrunning_loss, trainrunning_misclass_accuracy = optimality_measures(model, trainloader_whole, n_trainset, criterion, e)
        print(f"Training loss: {trainrunning_loss }")
        print("Training misclassification accuracy ", trainrunning_misclass_accuracy ,
              "and the gradient squared is", trainrunning_grad, " and epoch ", e)

        #Computing optimality measures w.r.t. the test set
        testrunning_grad, testrunning_loss, testrunning_misclass_accuracy = optimality_measures(model, testloader, n_testset, criterion, e)
        print(f"Test loss: {testrunning_loss}")
        print("Test misclassification accuracy ", testrunning_misclass_accuracy,
              "and the test gradient squared is", testrunning_grad, " and epoch ", e)
        TestAccuracy.append(testrunning_misclass_accuracy)
        TrainGradSquared.append(trainrunning_grad)

        #Updating the local models
        sdC = model.state_dict()
        for k in range(n_nodes):
            list_of_models[k].load_state_dict(sdC)

    returning_values = []
    returning_values.append(Machine_time)
    returning_values.append(TrainGradSquared)
    returning_values.append(TestAccuracy)
    returning_values.append(Time_FedGate)
    returning_values.append(TrainLoss)
    returning_values.append(list_of_models[0])
    pickle.dump(list_of_models[0], open("LR_Model_FedNova.dat", "wb"))
    return returning_values

def FedGate(model, trainloader_whole, node_trainloader, node_trainldr, testloader, epochs_FedGate, eta_FedGate, n_trainset, n_testset, b_size_FedGate, n_nodes, active_nodes, gamma, threshold_FedGate, doubling_factor, threshold_factor, List_of_speeds ):
    print("Running FedGate")
    if active_nodes != n_nodes:
        flag = 1
    else:
        flag =0
    # Defining models for each node and an optimizer
    TestAccuracy = []
    ActiveGradSquared = []
    TrainGradSquared =[]
    TrainLoss = []
    MachineTime = []
    list_of_models = []
    #optimizers = []
    deltas = []
    Deltas = []
    tempmodel2 = deepcopy(model)
    averg_Delta = tempmodel2.state_dict()
    for key in averg_Delta:
        averg_Delta[key] = averg_Delta[key] - averg_Delta[key]
    for i in range(n_nodes):
        tempmodel1 = deepcopy(model)
        tempmodel2 = deepcopy(model)
        list_of_models.append(tempmodel1)
        #optimizers.append(torch.optim.SGD(list_of_models[i].parameters(), lr=eta_FedGate))
        # Initializing Deltas and deltas
        deltas.append(list_of_models[i].state_dict())
        Deltas.append(tempmodel2.state_dict())
        for key in deltas[i]:
            deltas[i][key] = deltas[i][key] - deltas[i][key]
        for key in Deltas[i]:
            Deltas[i][key] = Deltas[i][key] - Deltas[i][key]
    # Define the loss
    criterion = nn.NLLLoss()
    epochs = epochs_FedGate
    counter = -1
    for e in range(epochs):
        counter += 1
        print("Epoch ", e, "counter ", counter, "threshold ", threshold_FedGate)
        #Add clock time
        if e == 0:
            Time_FedGate = []
            Time_FedGate.append(List_of_speeds[active_nodes - 1])
        else:
            var = Time_FedGate[e - 1] + List_of_speeds[active_nodes - 1]
            print("Time added ", List_of_speeds[active_nodes - 1])
            Time_FedGate.append(var)
        nodes_gradient_square = 0
        nodes_averg_loss = 0
        for j in range(active_nodes):
            for images, labels in node_trainloader[j]:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
                labels = labels.long()
                # Training pass
                output = list_of_models[j](images)
                # Computing Misclassification
                loss = criterion(output, labels)
                nodes_averg_loss += loss.item()
                #optimizers[j].zero_grad()
                list_of_models[j].zero_grad()
                loss.backward()
                # Manual Optimization Scheme
                with torch.no_grad():
                    for p in list_of_models[j].parameters():
                        nodes_gradient_square += (torch.norm(p.grad))**2
                        new_val = p - eta_FedGate * p.grad
                        p.copy_(new_val)
                sd = list_of_models[j].state_dict()
                for key in sd:
                    sd[key] = sd[key] + eta_FedGate * deltas[j][key]
                list_of_models[j].load_state_dict(sd)
        # Updating Deltas and averg Delta
        for key in averg_Delta:
            averg_Delta[key] -= averg_Delta[key]
        sdA = model.state_dict()
        for j in range(active_nodes):
            sdB = list_of_models[j].state_dict()
            for key in Deltas[j]:
                Deltas[j][key] = sdA[key] - sdB[key]
                averg_Delta[key] += Deltas[j][key] / active_nodes
        # Updating deltas
        for j in range(active_nodes):
            for key in deltas[j]:
                deltas[j][key] += (1 / (10 * eta_FedGate)) * (Deltas[j][key] - averg_Delta[key])
        # Updating the average model
        sd = model.state_dict()
        for key in sd:
            sd[key] -= gamma * averg_Delta[key]
        model.load_state_dict(sd)

        #computing optimality measures w.r.t. the active training set
        active_trainrunning_grad, active_trainrunning_loss, active_trainrunning_misclass_accuracy = 0, 0, 0
        """
        for j in range(active_nodes):
            trainrunning_grad, trainrunning_loss, trainrunning_misclass_accuracy = optimality_measures(model, node_trainldr[j], int(n_trainset/n_nodes), criterion, e)
            active_trainrunning_grad += trainrunning_grad/active_nodes
            active_trainrunning_loss += trainrunning_loss/active_nodes
            active_trainrunning_misclass_accuracy += trainrunning_misclass_accuracy/active_nodes
        print("Active loss: ", active_trainrunning_loss)
        print("Active misclass accuracy ", active_trainrunning_misclass_accuracy, "the active gradientis", active_trainrunning_grad, " and epoch ", e)
        """
        #computing optimality measures w.r.t. the training set
        trainrunning_grad, trainrunning_loss, trainrunning_misclass_accuracy = optimality_measures(model, trainloader_whole, n_trainset, criterion, e)
        print(f"Training loss: {trainrunning_loss }")
        print("Training misclassification accuracy ", trainrunning_misclass_accuracy ,
              "and the gradient squared is", trainrunning_grad, " and epoch ", e)
        #computing optimality measures w.r.t. the test set
        testrunning_grad, testrunning_loss, testrunning_misclass_accuracy = optimality_measures(model, testloader, n_testset, criterion, e)
        print(f"Test loss: {testrunning_loss }")
        print("Test misclassification accuracy ", testrunning_misclass_accuracy ,
              "and the gradient squared is", testrunning_grad, " and epoch ", e)

        ActiveGradSquared.append(active_trainrunning_grad)
        TrainGradSquared.append(trainrunning_grad)
        TestAccuracy.append(testrunning_misclass_accuracy)
        TrainLoss.append(trainrunning_loss)
        if e>49:
            if e==50:
                threshold_FedGate = TrainGradSquared[e]
            if counter%50==0 or TrainGradSquared[e]<threshold_FedGate:
                counter = 0
                if active_nodes==n_nodes:
                    pass
                else:
                    if active_nodes<=(n_nodes/2):
                        eta_FedGate = eta_FedGate / doubling_factor
                        gamma = gamma * doubling_factor
                        threshold_FedGate = threshold_FedGate / threshold_factor
                    active_nodes = min(n_nodes, 2 * active_nodes)
                    print("!!!!!!!!Doubling to ", active_nodes, "at epoch!!!!!!! ", e)
                    for i in range(n_nodes):
                        for key in deltas[i]:
                            deltas[i][key] = deltas[i][key] - deltas[i][key]
                        for key in Deltas[i]:
                            Deltas[i][key] = Deltas[i][key] - Deltas[i][key]


        # updating the models
        sdC = model.state_dict()
        for k in range(active_nodes):
            list_of_models[k].load_state_dict(sdC)

    returning_values = []
    returning_values.append(ActiveGradSquared)
    returning_values.append(TrainGradSquared)
    returning_values.append(TestAccuracy)
    returning_values.append(Time_FedGate)
    returning_values.append(TrainLoss)
    returning_values.append(list_of_models[0])
    if flag == 0:
        pickle.dump(list_of_models[0], open("LR_Model_FedGATE.dat", "wb"))
    else:
        pickle.dump(list_of_models[0], open("LR_FLANP.dat", "wb"))
    return returning_values

def FedGate_clusters(model, trainloader_whole, node_trainloader, testloader, epochs_FedGate, eta_FedGate, n_trainset, n_testset, b_size_FedGate, n_nodes, active_nodes, gamma, threshold_FedGate, doubling_factor, threshold_factor, List_of_speeds, n_clusters ):
    print("Running FedGate cluster")
    # Active nodes for each cluster
    active_nodes = active_nodes*n_clusters
    # Defining models for each node and an optimizer
    TestAccuracy = []
    ActiveGradSquared = []
    TrainGradSquared =[]
    TrainLoss = []
    MachineTime = []
    list_of_models = []
    #optimizers = []
    deltas = []
    Deltas = []
    tempmodel2 = deepcopy(model)
    averg_Delta = tempmodel2.state_dict()
    for key in averg_Delta:
        averg_Delta[key] = averg_Delta[key] - averg_Delta[key]
    for i in range(n_nodes):
        tempmodel1 = deepcopy(model)
        tempmodel2 = deepcopy(model)
        list_of_models.append(tempmodel1)
        #optimizers.append(torch.optim.SGD(list_of_models[i].parameters(), lr=eta_FedGate))
        # Initializing Deltas and deltas
        deltas.append(list_of_models[i].state_dict())
        Deltas.append(tempmodel2.state_dict())
        for key in deltas[i]:
            deltas[i][key] = deltas[i][key] - deltas[i][key]
        for key in Deltas[i]:
            Deltas[i][key] = Deltas[i][key] - Deltas[i][key]
    # Define the loss
    criterion = nn.NLLLoss()
    epochs = epochs_FedGate
    counter = -1
    for e in range(epochs):
        counter += 1
        print("Epoch ", e, "counter ", counter, "threshold ", threshold_FedGate)
        #Add clock time
        if e == 0:
            Time_FedGate = []
            max_speed = 0
            for z in range(n_clusters):
                if List_of_speeds[z][int(active_nodes/n_clusters)-1]>max_speed:
                    max_speed = List_of_speeds[z][int(active_nodes/n_clusters)-1]
            Time_FedGate.append(max_speed)
        else:
            max_speed = 0
            for z in range(n_clusters):
                if List_of_speeds[z][int(active_nodes/n_clusters) - 1] > max_speed:
                    max_speed = List_of_speeds[z][int(active_nodes/n_clusters) - 1]
            var = Time_FedGate[e - 1] + max_speed
            print("Time added ", max_speed)
            Time_FedGate.append(var)
        nodes_gradient_square = 0
        nodes_averg_loss = 0
        for j in range(active_nodes):
            for images, labels in node_trainloader[j%n_clusters][math.floor(j/n_clusters)]:
                #print("In active node ", j, "we have labels ", labels)
                #input("Waiting")
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
                labels = labels.long()
                # Training pass
                output = list_of_models[j](images)
                # Computing Misclassification
                loss = criterion(output, labels)
                nodes_averg_loss += loss.item()
                #optimizers[j].zero_grad()
                list_of_models[j].zero_grad()
                loss.backward()
                # Manual Optimization Scheme
                with torch.no_grad():
                    for p in list_of_models[j].parameters():
                        nodes_gradient_square += (torch.norm(p.grad))**2
                        new_val = p - eta_FedGate * p.grad
                        p.copy_(new_val)
                sd = list_of_models[j].state_dict()
                for key in sd:
                    sd[key] = sd[key] + eta_FedGate * deltas[j][key]
                list_of_models[j].load_state_dict(sd)
        # Updating Deltas and averg Delta
        for key in averg_Delta:
            averg_Delta[key] -= averg_Delta[key]
        sdA = model.state_dict()
        for j in range(active_nodes):
            sdB = list_of_models[j].state_dict()
            for key in Deltas[j]:
                Deltas[j][key] = sdA[key] - sdB[key]
                averg_Delta[key] += Deltas[j][key] / active_nodes
        # Updating deltas
        for j in range(active_nodes):
            for key in deltas[j]:
                deltas[j][key] += (1 / (10 * eta_FedGate)) * (Deltas[j][key] - averg_Delta[key])
        # Updating the average model
        sd = model.state_dict()
        for key in sd:
            sd[key] -= gamma * averg_Delta[key]
        model.load_state_dict(sd)

        """
        #computing optimality measures w.r.t. the active training set
        active_trainrunning_grad, active_trainrunning_loss, active_trainrunning_misclass_accuracy = 0, 0, 0
        for j in range(active_nodes):
            trainrunning_grad, trainrunning_loss, trainrunning_misclass_accuracy = optimality_measures(model, node_trainldr[j], int(n_trainset/n_nodes), criterion, e)
            active_trainrunning_grad += trainrunning_grad/active_nodes
            active_trainrunning_loss += trainrunning_loss/active_nodes
            active_trainrunning_misclass_accuracy += trainrunning_misclass_accuracy/active_nodes
        print("Active loss: ", active_trainrunning_loss)
        print("Active misclass accuracy ", active_trainrunning_misclass_accuracy, "the active gradientis", active_trainrunning_grad, " and epoch ", e)
        """

        #computing optimality measures w.r.t. the training set
        trainrunning_grad, trainrunning_loss, trainrunning_misclass_accuracy = optimality_measures(model, trainloader_whole, n_trainset, criterion, e)
        print(f"Training loss: {trainrunning_loss }")
        print("Training misclassification accuracy ", trainrunning_misclass_accuracy ,
              "and the gradient squared is", trainrunning_grad, " and epoch ", e)
        #computing optimality measures w.r.t. the test set
        testrunning_grad, testrunning_loss, testrunning_misclass_accuracy = optimality_measures(model, testloader, n_testset, criterion, e)
        print(f"Test loss: {testrunning_loss }")
        print("Test misclassification accuracy ", testrunning_misclass_accuracy ,
              "and the gradient squared is", testrunning_grad, " and epoch ", e)

        #ActiveGradSquared.append(active_trainrunning_grad)
        ActiveGradSquared.append(-3)

        TrainGradSquared.append(trainrunning_grad)
        TestAccuracy.append(testrunning_misclass_accuracy)
        TrainLoss.append(trainrunning_loss)
        if e>149:
            if e==150:
                threshold_FedGate = TrainGradSquared[e]/1000000000
            if counter%150==0 or TrainGradSquared[e]<threshold_FedGate:
                counter = 0
                if active_nodes==n_nodes:
                    pass
                else:
                    if active_nodes<=(n_nodes/2):
                        eta_FedGate = eta_FedGate / doubling_factor
                        gamma = gamma * doubling_factor
                        threshold_FedGate = threshold_FedGate / threshold_factor
                    active_nodes = min(n_nodes, 2 * active_nodes)
                    print("!!!!!!!!Doubling to ", active_nodes, "at epoch!!!!!!! ", e)
                    for i in range(n_nodes):
                        for key in deltas[i]:
                            deltas[i][key] = deltas[i][key] - deltas[i][key]
                        for key in Deltas[i]:
                            Deltas[i][key] = Deltas[i][key] - Deltas[i][key]


        # updating the models
        sdC = model.state_dict()
        for k in range(n_nodes):
            list_of_models[k].load_state_dict(sdC)

    returning_values = []
    returning_values.append(ActiveGradSquared)
    returning_values.append(TrainGradSquared)
    returning_values.append(TestAccuracy)
    returning_values.append(Time_FedGate)
    returning_values.append(TrainLoss)
    return returning_values

def Plotter():
    restrictbound = 0.10
    restrict = 1000
    listofthings = pickle.load(open("Femnist_FLANP_Clusters.dat", "rb"))
    listofthings_2 = pickle.load(open("Femnist_FedGATE_Clusters.dat", "rb"))
    listofthings_3 = pickle.load(open("Femnist_FedAvg_Clusters.dat", "rb"))
    listofthings_4 = pickle.load(open("Femnist_FedNova_Clusters.dat", "rb"))
    #listofthings_5 = pickle.load(open("exp_MNIST_nn_FLANP_FedAvg_Allnodes_faster_nodes_new_try.dat", "rb"))
    #listofthings_6 = pickle.load(open("testing_exp_MNIST_nn_FLANP_FedNova_last.dat", "rb"))


    listofthings[2] = [1-x for x in listofthings[2]]
    listofthings_2[2] = [1 - x for x in listofthings_2[2]]
    listofthings_3[2] = [1 - x for x in listofthings_3[2]]
    listofthings_4[2] = [1 - x for x in listofthings_4[2]]
    #listofthings_5[2] = [1 - x for x in listofthings_5[2]]
    #listofthings_6[2] = [1 - x for x in listofthings_6[2]]


    listofthings[2].insert(0, 0.87)
    listofthings_2[2].insert(0, 0.87)
    listofthings_3[2].insert(0, 0.87)
    listofthings_4[2].insert(0, 0.87)
    #listofthings_5[2].insert(0, 0.87)
    #listofthings_6[2].insert(0, 0.87)


    listofthings[3].insert(0, 0.00)
    listofthings_2[3].insert(0, 0.00)
    listofthings_3[3].insert(0, 0.00)
    listofthings_4[3].insert(0, 0.00)
    #listofthings_5[3].insert(0, 0.00)
    #listofthings_6[3].insert(0, 0.00)


    for i in range(len(listofthings[2])):
        if listofthings[2][i] < restrictbound :
            restrict_1 = i
            #print("The time limit for first is ", restrict_1)
            break

    restrict_1 = 1000

    for i in range(len(listofthings_2[2])):
        if listofthings_2[2][i] < restrictbound :
            restrict_2 = i
            #print("The time limit for second is ", restrict_2)
            break

    restrict_2 = 1000
    for i in range(len(listofthings_3[2])):
        if listofthings_3[2][i] < restrictbound:
            restrict = i
            # print("The time limit for second is ", restrict_4)
            break
    restrict_4 = 1000
    for i in range(len(listofthings_4[2])):
        if listofthings_4[2][i] < restrictbound :
            restrict_4 = i
            #print("The time limit for second is ", restrict_4)
            break

    #restrict_2 = 1000
    """
    for i in range(len(listofthings_5[2])):
        if listofthings_5[2][i] < restrictbound :
            restrict_5 = i
            #print("The time limit for second is ", restrict_4)
            break

    #restrict_2 = 1000

    for i in range(len(listofthings_6[2])):
        if listofthings_6[2][i] < restrictbound :
            restrict_6 = i
            #print("The time limit for second is ", restrict_4)
            break
    """
    #restrict_2 = 1000

    plt.plot(range(len(listofthings[2][0:restrict_1])), listofthings[2][0:restrict_1], marker='', linestyle='-', label='FLANP with FedGATE', linewidth = '4')
    plt.plot(range(len(listofthings_2[2][0:restrict_2])), listofthings_2[2][0:restrict_2], marker='>', linestyle='-', label='FedGATE', linewidth = '4')
    plt.plot(range(len(listofthings_3[2][0:restrict])), listofthings_3[2][0:restrict], marker='', linestyle='-', label='FedAvg', linewidth = '4')
    plt.plot(range(len(listofthings_4[2][0:restrict_4])), listofthings_4[2][0:restrict_4], marker='', linestyle='-', label='FedNova', linewidth = '4')
    #plt.plot(range(len(listofthings_5[2][0:restrict_5])), listofthings_5[2][0:restrict_5], marker='', linestyle='-', label='FLANP with FedAvg', linewidth = '4')
    #plt.plot(range(len(listofthings_6[2][0:restrict_6])), listofthings_6[2][0:restrict_6], marker='>', linestyle='-', label='FLANP with FedNova', linewidth = '4')


    plt.xlabel('Communication Rounds', fontsize=16)
    plt.ylabel('Misclassification Error', fontsize=16)
    plt.legend()
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title("Misclassification to Communication Rounds", fontsize=18)
    plt.grid()
    #plt.yscale('log')
    plt.savefig('equal_testing_Exp_Missclassification_Error_to_Communication_FLANP_with_FedAvg.pdf')
    plt.show()
    input('Done with phase I and II')

    plt.plot(listofthings[3][0:restrict_1], listofthings[2][0:restrict_1], marker='', linestyle='-', label='FLANP with FedGATE', linewidth = '4')
    plt.plot(listofthings_2[3][0:restrict_2], listofthings_2[2][0:restrict_2], marker='>', linestyle='-', label='FedGATE', linewidth = '4')
    plt.plot(listofthings_3[3][0:restrict], listofthings_3[2][0:restrict], marker='', linestyle='-', label='FedAvg', linewidth = '4')
    plt.plot(listofthings_4[3][0:restrict_4], listofthings_4[2][0:restrict_4], marker='', linestyle='-', label='FedNova', linewidth = '4')
    #plt.plot(listofthings_5[3][0:restrict_5], listofthings_5[2][0:restrict_5], marker='', linestyle='-', label='FLANP with FedAvg', linewidth = '4')
    #plt.plot(listofthings_6[3][0:restrict_6], listofthings_6[2][0:restrict_6], marker='>', linestyle='-', label='FLANP with FedNova', linewidth = '4')


    plt.xlabel('Wall-Clock Time', fontsize=16)
    plt.ylabel('Misclassification Error', fontsize=16)
    plt.legend()
    plt.legend(fontsize=15)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    #plt.xscale('log')
    plt.savefig('equal_testing_Exp_Global_Loss_to_Clock_Time_FLANP_with_FedAvg.pdf')
    plt.show()
    input('Done with phase I and II')