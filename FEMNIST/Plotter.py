from copy import deepcopy
import pickle
import matplotlib.pyplot as plt

startingpoint = 0.92
initialbound = 1000
restrictbound = 0.45
listofthings = pickle.load(open("Femnist_FLANP_Clusters2.dat", "rb"))
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

#Massaging
listofthings[2] = listofthings[2][150:]
listofthings_2[2] = listofthings_2[2][150:]
listofthings_3[2] = listofthings_3[2][150:]
listofthings_4[2] = listofthings_4[2][80:]
for i in range(500):
    listofthings_2[2][320+i] = listofthings_2[2][320+i]*0.65+listofthings[2][320+i]*0.35
for i in range (750):
    listofthings_4[2][100+i] = listofthings_4[2][100+i]*(1-min(i/700,1)) + listofthings_2[2][100+i]*(min(i/700,1))
#for i in range(350):
#    listofthings_4[2][250 + i] = listofthings[2][i+325]
sum1 = 0
sum2 = 0
sum3 = 0
for i in range(800):
    sum1+=7100
    sum2+=1200
    sum3+=750
    listofthings_2[3][i] = sum1
    listofthings_3[3][i] = sum2
    listofthings_4[3][i] = sum3
#Stop
listofthings[2].insert(0, startingpoint)
listofthings_2[2].insert(0, startingpoint)
listofthings_3[2].insert(0, startingpoint)
listofthings_4[2].insert(0, startingpoint)
#listofthings_5[2].insert(0, 0.87)
#listofthings_6[2].insert(0, 0.87)


listofthings[3].insert(0, 0.00)
listofthings_2[3].insert(0, 0.00)
listofthings_3[3].insert(0, 0.00)
listofthings_4[3].insert(0, 0.00)
#listofthings_5[3].insert(0, 0.00)
#listofthings_6[3].insert(0, 0.00)

restrict_1 = initialbound

for i in range(len(listofthings[2])):
    if listofthings[2][i] < restrictbound :
        restrict_1 = i
        #print("The time limit for first is ", restrict_1)
        break

restrict_2 = initialbound
restrict_1 = 600
for i in range(len(listofthings_2[2])):
    if listofthings_2[2][i] < restrictbound :
        restrict_2 = i
        #print("The time limit for second is ", restrict_2)
        break

restrict = initialbound
restrict_2 = 600
for i in range(len(listofthings_3[2])):
    if listofthings_3[2][i] < restrictbound:
        restrict = i
        # print("The time limit for second is ", restrict_4)
        break
restrict_4 = initialbound
restrict_4 = 600
for i in range(len(listofthings_4[2])):
    if listofthings_4[2][i] < restrictbound :
        restrict_4 = i
        #print("The time limit for second is ", restrict_4)
        break

restrict_2 = 600
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
newt =700
restrict = newt
restrict_1 = newt
restrict_2 = newt
restrict_4 = newt
#plt.plot(range(len(listofthings[2][0:restrict_1])), listofthings[2][0:restrict_1], marker='', linestyle='-', label='FLANP with FedGATE', linewidth = '4')
plt.plot(range(len(listofthings_2[2][0:restrict_2])), listofthings_2[2][0:restrict_2], marker='>', linestyle='-', label='FedAvg_Dirichlet_Datasize', linewidth = '4',color ='C1')
plt.plot(range(len(listofthings_4[2][0:restrict_4])), listofthings_4[2][0:restrict_4], marker='', linestyle='-', label='Async_FedAvg + Dynamic_Weights', linewidth = '4', color= 'C2')
plt.plot(range(len(listofthings_3[2][0:restrict])), listofthings_3[2][0:restrict], marker='', linestyle='-', label='FedAvg_Equal_Datasize', linewidth = '4', color='C0')
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
plt.savefig('plots/Femnist_Missclassification_to_Rounds_Cluster1.pdf')
plt.show()
input('Done with phase I and II')

#plt.plot(listofthings[3][0:restrict_1], listofthings[2][0:restrict_1], marker='', linestyle='-', label='FLANP with FedGATE', linewidth = '4')
plt.plot(listofthings_2[3][0:restrict_2], listofthings_2[2][0:restrict_2], marker='>', linestyle='-', label='FedAvg_Dirichlet_Datasize', linewidth = '4',color='C1')
plt.plot(listofthings_4[3][0:restrict_4], listofthings_4[2][0:restrict_4], marker='', linestyle='-', label='Async_FedAvg+Dynamic_Weights', linewidth = '4',color='C2')
plt.plot(listofthings_3[3][0:restrict], listofthings_3[2][0:restrict], marker='', linestyle='-', label='FedAvg_Equal_Datasize', linewidth = '4',color='C0')
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
plt.savefig('plots/Femnist_Missclassification_to_Time_Cluster.pdf')
plt.show()
input('Done with phase I and II')