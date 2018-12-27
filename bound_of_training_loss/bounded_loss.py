import numpy as np
from sklearn import svm, datasets
from Data_dist import data_partition, kmeans_partition
import Edge_node as Ed
import sklearn
import random_select as Rs
import Central_node as Ce
from sklearn.utils import shuffle
from sklearn import preprocessing
import copy
import matplotlib.pyplot as plt
from plot_utility import SVM_plot
from Data_load import read_UCI_data, read_2D


Train_data, Train_label = read_UCI_data(0)

C = 3.0
gamma = 0.1
Global_model = svm.SVC(C = C, kernel = 'rbf', gamma=gamma, tol=1e-6)

Global_model.fit(Train_data, Train_label)
Global_loss = Ce.training_loss(Global_model)
# print(Ce.training_loss(Global_model))

# Edge_node_n = 2
# Edge_data, Edge_label, Global_index = \
#     data_partition(Train_data, Train_label, Edge_node_n)
#
# Edge_loss = np.zeros(Edge_node_n)
# Edge_upper_loss = np.zeros(Edge_node_n)
#
# for i in range(Edge_node_n):
#     local_model = Ed.local_train(Edge_data[i], Edge_label[i], C, gamma, 'rbf')
#     Edge_loss[i] = Ce.training_loss(local_model)
#
#     upper_model = Ed.local_train(Edge_data[i], Edge_label[i], C*2*Edge_node_n, gamma, 'rbf')
#     Edge_upper_loss[i] = \
#         (1/(2*Edge_node_n)) * Ce.training_loss(upper_model)
#
# print('Upper bound:')
# print(np.sum(Edge_upper_loss))
# print('True value:')
# print(Global_loss)
# print('Lower bound:')
# print(np.sum(Edge_loss))

for i in range(2,30):
    Edge_node_n = i
    Edge_data, Edge_label, Global_index = \
        data_partition(Train_data, Train_label, Edge_node_n)

    Edge_lower_loss = np.zeros(Edge_node_n)
    Edge_upper_loss = np.zeros(Edge_node_n)

    for i in range(Edge_node_n):
        # lower bound of training loss for global model
        local_model = Ed.local_train(Edge_data[i], Edge_label[i], C, gamma, 'rbf')
        Edge_lower_loss[i] = Ce.training_loss(local_model)

        # upper bound of training loss for global model
        upper_model = Ed.local_train(Edge_data[i], Edge_label[i], C * 2 * Edge_node_n, gamma, 'rbf')
        Edge_upper_loss[i] = \
            (1 / (2 * Edge_node_n)) * Ce.training_loss(upper_model)

    if (Global_loss >= np.sum(Edge_lower_loss) and Global_loss <= np.sum(Edge_upper_loss)):
        print('The principle is right for Edge_node_n = %d' %(Edge_node_n))
    else:
        print('The principle is false for Edge_node_n = %d' %(Edge_node_n))


