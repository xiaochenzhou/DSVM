from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np
import numpy as np
from sklearn import svm, datasets
from Data_dist import data_partition, kmeans_partition
import Edge_node as Ed
import sklearn
import Central_node as Ce
from sklearn.utils import shuffle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from plot_utility import SVM_plot
from imblearn.over_sampling import SMOTE, ADASYN
from random_select import random_select, k_means_random_select
from collections import Counter

X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print (np.size (y, axis = 0))
print (y)
print('Original dataset shape %s' % Counter(y))
X_res, y_res = SMOTE(random_state=42).fit_sample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
print(y)
print (X)

Train_data = np.load("Train_data_5D.npy")
Train_label = np.load("Train_label_5D.npy")
Train_label = 2 * Train_label - 1

Train_data, Train_label = shuffle ( Train_data, Train_label, random_state = 1)

Test_data = Train_data[8000:10000]
Test_label = Train_label[8000:10000]
print ("Test size is %s" %(np.size (Test_label)))
Train_data = Train_data[:8000]
Train_label = Train_label[:8000]

print ("global size is %s" %(np.size (Train_label)))

Train_label += 1
print (Train_label)
