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

def read_UCI_data(num):
    if (num == 0):
        Training_set = sklearn.datasets.load_svmlight_file('skin_nonskin.txt')
    elif (num == 1):
        Training_set = sklearn.datasets.load_svmlight_file('phishing.txt')
    elif (num == 2):
        Training_set = sklearn.datasets.load_svmlight_file('mushrooms.txt')
    elif (num == 3):
        Training_set = sklearn.datasets.load_svmlight_file('ijcnn1.tr')
    elif (num == 4):
        Training_set = sklearn.datasets.load_svmlight_file('mnist.scale')
    else:
        raise ValueError


    Train_label_t = Training_set[1]
    Train_label_t = (2 * (Train_label_t % 2)) - 1
    Train_data_t = np.array(Training_set[0].todense())
    Train_data_t = preprocessing.normalize(Train_data_t)

    Train_data_s, Train_label_s = shuffle(Train_data_t, Train_label_t, random_state = 0)

    Train_label = Train_label_s[:10000]
    Train_data = Train_data_s[:10000, :]

    return Train_data, Train_label

def read_2D():
    Train_data = np.load("Train_data_1.npy")
    Train_label = np.load("Train_label_1.npy")
    Train_label = 2 * Train_label - 1

    return Train_data, Train_label