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

from random_select import random_select, k_means_random_select


def array_compare(element, array):
    true_table = (array == element)

    if (np.size(element) in np.sum(true_table, axis=1)):
        return True
    else:
        return False

def SV_compare(old_SV, new_SV):
    if (np.size(old_SV, axis=0) != np.size(new_SV, axis=0)):
        return False
    for i in range(np.size(new_SV, axis=0)):
        if  not array_compare(new_SV[i], old_SV):
            return False

    return True
def SV_diff_count(global_SV, dis_SV):
    count = 0
    if(np.size(global_SV, axis= 0)>=np.size(dis_SV, axis=0)):
        for i in range(np.size(dis_SV, axis=0)):
            if(not array_compare(dis_SV[i], global_SV)):
                count = count + 1
        count = count + np.size(global_SV, axis= 0) - np.size(dis_SV, axis=0)
    else:
        for i in range(np.size(global_SV, axis=0)):
            if(not array_compare(global_SV[i], dis_SV)):
                count = count + 1
        count = count + np.size(dis_SV, axis=0) - np.size(global_SV, axis=0)
    return count

def training_iteration(Edge_node_n, Edge_data, Edge_label, Upload_support_vector_,
                       Collect_support_vector_, Collect_label, C, gamma, Upload_label):
    test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    test.fit(Collect_support_vector_, Collect_label)

    new_support_vector_plus = []
    new_support_plus = []
    new_support_vector_minus = []
    new_support_minus = []

    for i in range(Edge_node_n):
        for j in range(np.size(Collect_support_vector_, axis=0)):
            if (not array_compare(Collect_support_vector_[j], Edge_data[i])):
                Edge_data[i] = np.concatenate((Edge_data[i], [Collect_support_vector_[j]]), axis=0)
                Edge_label[i] = np.concatenate((Edge_label[i], [Collect_label[j]]), axis=0)

        local_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        local_model.fit(Edge_data[i], Edge_label[i])

        support_, support_vectors_, n_support_ = Ed.local_support(local_model)

        D_plus = np.multiply(Edge_label[i][support_[n_support_[0]:]],
                             local_model.decision_function(support_vectors_[n_support_[0]:, :]))
        D_minus = np.multiply(Edge_label[i][support_[:n_support_[0]]],
                              local_model.decision_function(support_vectors_[:n_support_[0], :]))

        S_plus_, SV_plus \
            = Ed.upload_sort(support_[n_support_[0]:], support_vectors_[n_support_[0]:, :], D_plus)

        S_minus_, SV_minus \
            = Ed.upload_sort(support_[:n_support_[0]], support_vectors_[:n_support_[0], :], D_minus)

        Distance_plus[i] = D_plus
        Distance_minus[i] = D_minus
        support_plus_[i] = S_plus_
        support_minus_[i] = S_minus_
        support_vectors_plus[i] = SV_plus
        support_vectors_minus[i] = SV_minus

        new_SV_plus = []
        new_SV_minus = []
        new_S_plus_ = []
        new_S_minus_ = []
        for j in range(np.size(SV_plus, axis=0)):
            if (not array_compare(SV_plus[j], Collect_support_vector_)):
                new_SV_plus.append(SV_plus[j])
                new_S_plus_.append(S_plus_[j])
        for j in range(np.size(SV_minus, axis=0)):
            if (not array_compare(SV_minus[j], Collect_support_vector_)):
                new_SV_minus.append(SV_minus[j])
                new_S_minus_.append(S_minus_[j])

        new_support_vector_plus.append(np.array(new_SV_plus))
        new_support_plus.append(np.array(new_S_plus_))
        new_support_vector_minus.append(np.array(new_SV_minus))
        new_support_minus.append(np.array(new_S_minus_))

    stop_flag = [[False, False] for i in range(Edge_node_n)]
    while (1):
        for i in range(Edge_node_n):
            k1 = new_support_vector_plus[i][:1]
            k2 = new_support_vector_minus[i][:1]

            if (np.size(k1) == 0 and np.size(k2) == 0 or (stop_flag[i][0] and stop_flag[i][1])):
                stop_flag[i][0] = True
                stop_flag[i][1] = True
                continue
            elif (np.size(k1) == 0 and np.size(k2) != 0):
                stop_flag[i][0] = True
                if(test.decision_function(k2)* Edge_label[i][new_support_minus[i][:1]]> 1.0):
                    stop_flag[i][1] = True
                    continue
                else:
                    Upload_edge_support_vector_ = k2
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = Edge_label[i][new_support_minus[i][:1]]

                    new_support_minus[i], new_support_vector_minus[i] \
                        = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

            elif (np.size(k1) != 0 and np.size(k2) == 0):
                stop_flag[i][1] = True
                if (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > 1.0):
                    stop_flag[i][0] = True
                    continue
                else:
                    Upload_edge_support_vector_ = k1
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = Edge_label[i][new_support_plus[i][:1]]

                    new_support_plus[i], new_support_vector_plus[i] \
                        = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

            else:
                if(test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > 1.0
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > 1.0):
                    stop_flag[i][0] = True
                    stop_flag[i][1] = True
                    continue
                elif (test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] > 1.0
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] <= 1.0):
                    stop_flag[i][0] = True
                    Upload_edge_support_vector_ = k2
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)
                    Upload_edge_label = Edge_label[i][new_support_minus[i][:1]]

                    new_support_minus[i], new_support_vector_minus[i] \
                        = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

                elif(test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] <= 1.0
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] > 1.0):
                    stop_flag[i][1] = True
                    Upload_edge_support_vector_ = k1
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = Edge_label[i][new_support_plus[i][:1]]

                    new_support_plus[i], new_support_vector_plus[i] \
                        = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)

                elif(test.decision_function(k1) * Edge_label[i][new_support_plus[i][:1]] <= 1.0
                        and test.decision_function(k2) * Edge_label[i][new_support_minus[i][:1]] <= 1.0):
                    Upload_edge_support_vector_ = np.concatenate((k1, k2), axis=0)
                    Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_),
                                                               axis=0)

                    Upload_edge_label = np.concatenate(
                        (Edge_label[i][new_support_plus[i][:1]], Edge_label[i][new_support_minus[i][:1]]), axis=0)

                    new_support_plus[i], new_support_vector_plus[i] \
                        = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
                    new_support_minus[i], new_support_vector_minus[i] \
                        = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])

                    Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)


        Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
        Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))

        if (Edge_node_n > 2):
            for j in range(2, Edge_node_n):
                Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
                Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))

        old_support_vectors_ = test.support_vectors_

        test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        test.fit(Collect_support_vector_, Collect_label)

        new_support_vectors_ = test.support_vectors_

        if (False not in np.array(stop_flag).reshape((-1))):
            break


    print(np.size(Collect_label, axis=0))
    print(test.n_support_)

    return new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label


Train_data = np.load("Train_data_5D.npy")
Train_label = np.load("Train_label_5D.npy")
Train_label = 2 * Train_label - 1

print ("global size is %s" %(np.size (Train_label)))

# Training_set = sklearn.datasets.load_svmlight_file('german')
#
# Train_label_t = Training_set[1]
# Train_data_t = np.array(Training_set[0].todense())
# Train_data_t = preprocessing.normalize(Train_data_t)
#
# Train_data_s,Train_label_s = shuffle(Train_data_t, Train_label_t)
#
# Train_label = Train_label_s
# Train_data = Train_data_s


C = 3.0
gamma = 1.0 / 3.0

Global_model = svm.SVC(C = C, kernel = 'rbf', gamma=gamma)

Global_model.fit(Train_data, Train_label)

Only_SV_model = svm.SVC(C = C, kernel = 'rbf', gamma=gamma)
Only_SV_model.fit(Global_model.support_vectors_, Train_label[Global_model.support_])

print(Global_model.n_support_)
print(Only_SV_model.n_support_)

# SVM_plot(Train_data[:, 0], Train_data[:, 1], Train_label, Global_model)
# SVM_plot(Global_model.support_vectors_[:, 0], Global_model.support_vectors_[:, 1], Train_label[Global_model.support_], Only_SV_model)

Edge_node_n = 10
# Edge_data, Edge_label, Global_index = data_partition(Train_data, Train_label, Edge_node_n)
Edge_data, Edge_label, Global_index = kmeans_partition(Train_data, Train_label, Edge_node_n)
Distance_plus = []
Distance_minus = []
support_plus_ = []
support_minus_ = []
support_vectors_plus = []
support_vectors_minus = []

# this part is responsible for random select  ***********************

# local_additional_uploaded_node_data = np.array([ Edge_data[0] ])
additional_data =  Edge_data[0]
# local_additional_uploaded_node_label = np.array ( [Edge_label[0] ])
additional_label =  Edge_label[0]

#endcall

for i in range(Edge_node_n):
    local_model = Ed.local_train(Edge_data[i], Edge_label[i], C, gamma, 'rbf')

    # SVM_plot(Edge_data[i][:, 0], Edge_data[i][:, 1], Edge_label[i], local_model)
    support_, support_vectors_, n_support_ = Ed.local_support(local_model)

    D_plus = np.multiply(Edge_label[i][support_[n_support_[0]:]],
                          local_model.decision_function(support_vectors_[n_support_[0]:,:]))
    D_minus = np.multiply(Edge_label[i][support_[:n_support_[0]]],
                          local_model.decision_function(support_vectors_[:n_support_[0], :]))

    S_plus_, SV_plus\
        = Ed.upload_sort(support_[n_support_[0]:], support_vectors_[n_support_[0]:,:], D_plus)

    S_minus_, SV_minus\
        = Ed.upload_sort(support_[:n_support_[0]], support_vectors_[:n_support_[0],:], D_minus)

    Distance_plus.append(D_plus)
    Distance_minus.append(D_minus)
    support_plus_.append(S_plus_)
    support_minus_.append(S_minus_)
    support_vectors_plus.append(SV_plus)
    support_vectors_minus.append(SV_minus)

'''
 If uploading all the local support vectors
'''


local_support_vector = np.concatenate((support_vectors_plus[0], support_vectors_minus[0]),axis=0)
local_label_plus = np.ones(np.size(support_vectors_plus[0],axis=0))
local_label_minus = np.ones(np.size(support_vectors_minus[0],axis=0))*(-1)
local_label = np.concatenate((local_label_plus, local_label_minus),axis=0)



for i in range(1,Edge_node_n):
    local_support_vector = np.concatenate((local_support_vector, support_vectors_plus[i], support_vectors_minus[i]), axis=0)
    local_label_plus = np.ones(np.size(support_vectors_plus[i], axis=0))
    local_label_minus = np.ones(np.size(support_vectors_minus[i], axis=0)) * (-1)
    local_label = np.concatenate((local_label, local_label_plus, local_label_minus), axis=0)

All_upload_model = svm.SVC(C = C, kernel = 'rbf', gamma=gamma)
All_upload_model.fit(local_support_vector, local_label)

# SVM_plot(local_support_vector[:, 0], local_support_vector[:, 1], local_label, All_upload_model)
print(np.size(local_label, axis=0))
print(All_upload_model.n_support_)
'''
Central update
Outlier upload
'''


Updated_support_plus_ = []
Updated_support_minus_ = []
Updated_support_vectors_plus = []
Updated_support_vectors_minus = []

Upload_support_vector_ = []
Upload_label = []

# Upload outliers
Outlier_plus = np.zeros((Edge_node_n,1), dtype= np.int32)
Outlier_minus = np.zeros((Edge_node_n,1), dtype= np.int32)

for i in range(Edge_node_n):
    Outlier_plus[i] = np.sum((Distance_plus[i]<=0))
    Outlier_minus[i] = np.sum((Distance_minus[i]<=0))

    Outlier_n = np.max([Outlier_plus[i], Outlier_minus[i]])

    Upload_edge_support_vector_ = np.concatenate(
        (support_vectors_plus[i][:2*Outlier_n,:],support_vectors_minus[i][:2*Outlier_n,:]), axis = 0)
    Upload_support_vector_.append(Upload_edge_support_vector_)

    label_plus = Edge_label[i][support_plus_[i][:2*Outlier_n]]
    label_minus = Edge_label[i][support_minus_[i][:2*Outlier_n]]
    Upload_edge_label = np.concatenate((label_plus, label_minus))
    Upload_label.append(Upload_edge_label)

    US_plus_, USV_plus\
        = Ed.local_upload_outlier(support_plus_[i], support_vectors_plus[i], 2*Outlier_n)
    US_minus_, USV_minus\
        = Ed.local_upload_outlier(support_minus_[i], support_vectors_minus[i], 2*Outlier_n)


    Updated_support_plus_.append(US_plus_)
    Updated_support_minus_.append(US_minus_)
    Updated_support_vectors_plus.append(USV_plus)
    Updated_support_vectors_minus.append(USV_minus)

Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
if (Edge_node_n > 2):
    for j in range(2, Edge_node_n):
        Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
        Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))


for i in range(Edge_node_n):
    
# this part is responsible for random select

    additional_data, additional_label = random_select(10000, Edge_data[i], Edge_label[i], Collect_support_vector_)
    # local_additional_uploaded_node_data_temp, local_additional_uploaded_node_label_temp = k_means_random_select(5000, Edge_data[i], Edge_label[i], Collect_support_vector_)

    Upload_support_vector_[i] = np.concatenate ((Upload_support_vector_[i], additional_data), axis = 0)
    Upload_label[i] = np.concatenate ((Upload_label[i], additional_label), axis=0 )


    print("addational size is %s" %(np.size( additional_label )))
#end call

#this part is for random select 
# print (Collect_label)
# print (Collect_support_vector_)
# print (local_additional_uploaded_node_data)
# Collect_label = np.concatenate( (local_additional_uploaded_node_label, Collect_label), axis = 0  )
# Collect_support_vector_ = np.concatenate ( (local_additional_uploaded_node_data, Collect_support_vector_), axis = 0  )
#end call

Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
if (Edge_node_n > 2):
    for j in range(2, Edge_node_n):
        Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
        Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))

test=svm.SVC(C = C, kernel = 'rbf', gamma=gamma)
test.fit(Collect_support_vector_, Collect_label)
print ("begin")
print(np.size(Collect_label, axis=0))


# while((np.sum(test.n_support_)) == np.size(Collect_support_vector_, axis=0)):
# # while ((np.sum(test.n_support_)) == np.size(Collect_support_vector_, axis=0)):
#     for i in range(Edge_node_n):
#         k1= [Updated_support_vectors_plus[i][0]]
#         k2 = [Updated_support_vectors_minus[i][0]]
#         Upload_edge_support_vector_ = np.concatenate(
#             (k1, k2), axis=0)
#         Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_), axis=0)
#
#         Upload_edge_label = [1.0, -1.0]
#         Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)
#
#         Updated_support_plus_[i], Updated_support_vectors_plus[i] \
#             = Ed.local_upload(Updated_support_plus_[i], Updated_support_vectors_plus[i])
#         Updated_support_minus_[i], Updated_support_vectors_minus[i] \
#             = Ed.local_upload(Updated_support_minus_[i], Updated_support_vectors_minus[i])
#
#     Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
#     Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
#
#     if(Edge_node_n > 2):
#         for j in range(2,Edge_node_n):
#             Collect_support_vector_ = np.concatenate((Collect_support_vector_,Upload_support_vector_[j]))
#             Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))
#     test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
#     test.fit(Collect_support_vector_, Collect_label)
#
#
# print(np.size(Collect_label, axis=0))
# print(test.n_support_)

ite_count = 0
print ("ite_count = %s" %(ite_count))

old_support_vectors_ = test.support_vectors_
new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label = training_iteration(Edge_node_n, Edge_data, Edge_label, Upload_support_vector_,
                       Collect_support_vector_, Collect_label, C, gamma, Upload_label)
ite_count += 1



while(not SV_compare(old_support_vectors_, new_support_vectors_)):
    print ("ite_count = %s" %(ite_count)) 
    old_support_vectors_ = new_support_vectors_
    new_support_vectors_, Upload_support_vector_, Collect_support_vector_, Collect_label = \
        training_iteration(Edge_node_n, Edge_data, Edge_label, Upload_support_vector_,
                       Collect_support_vector_, Collect_label, C, gamma, Upload_label)
    ite_count += 1

print ("ite_count = %s" %(ite_count)) 


print(SV_diff_count(Global_model.support_vectors_ ,new_support_vectors_))



# # new iteration of local training
# new_support_vector_plus = []
# new_support_plus = []
# new_support_vector_minus = []
# new_support_minus = []
#
# for i in range(Edge_node_n):
#     for j in range(np.size(Collect_support_vector_, axis=0)):
#         if(Collect_support_vector_[j] not in Edge_data[i]):
#             Edge_data[i] = np.concatenate((Edge_data[i], [Collect_support_vector_[j]]), axis=0)
#             Edge_label[i] = np.concatenate((Edge_label[i], [Collect_label[j]]), axis=0)
#
#     local_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
#     local_model.fit(Edge_data[i], Edge_label[i])
#     # SVM_plot(Edge_data[i][:, 0], Edge_data[i][:, 1], Edge_label[i], local_model)
#
#     support_, support_vectors_, n_support_ = Ed.local_support(local_model)
#
#     D_plus = np.multiply(Edge_label[i][support_[n_support_[0]:]],
#                           local_model.decision_function(support_vectors_[n_support_[0]:,:]))
#     D_minus = np.multiply(Edge_label[i][support_[:n_support_[0]]],
#                           local_model.decision_function(support_vectors_[:n_support_[0], :]))
#
#     S_plus_, SV_plus\
#         = Ed.upload_sort(support_[n_support_[0]:], support_vectors_[n_support_[0]:,:], D_plus)
#
#     S_minus_, SV_minus\
#         = Ed.upload_sort(support_[:n_support_[0]], support_vectors_[:n_support_[0],:], D_minus)
#
#     Distance_plus[i] = D_plus
#     Distance_minus[i] = D_minus
#     support_plus_[i] = S_plus_
#     support_minus_[i] = S_minus_
#     support_vectors_plus[i] = SV_plus
#     support_vectors_minus[i] = SV_minus
#
#     new_SV_plus = []
#     new_SV_minus = []
#     new_S_plus_ = []
#     new_S_minus_ = []
#     for j in range(np.size(SV_plus, axis=0)):
#         if(SV_plus[j] not in Collect_support_vector_):
#             new_SV_plus.append(SV_plus[j])
#             new_S_plus_.append(S_plus_[j])
#     for j in range(np.size(SV_minus, axis=0)):
#         if(SV_minus[j] not in Collect_support_vector_):
#             new_SV_minus.append(SV_minus[j])
#             new_S_minus_.append(S_minus_[j])
#
#     new_support_vector_plus.append(np.array(new_SV_plus))
#     new_support_plus.append(np.array(new_S_plus_))
#     new_support_vector_minus.append(np.array(new_SV_minus))
#     new_support_minus.append(np.array(new_S_minus_))
#
# # print(new_support_vector_plus)
# # print(new_support_vector_minus)
#
# while(1):
#     for i in range(Edge_node_n):
#         k1 = new_support_vector_plus[i][:1]
#         k2 = new_support_vector_minus[i][:1]
#
#         if (np.size(k1) == 0 and np.size(k2) == 0):
#             break
#         elif (np.size(k1) == 0 and np.size(k2) != 0):
#             Upload_edge_support_vector_ = k2
#             Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_), axis=0)
#
#             Upload_edge_label = Edge_label[i][new_support_minus[i][:1]]
#
#             new_support_minus[i], new_support_vector_minus[i] \
#                 = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
#
#         elif (np.size(k1) != 0 and np.size(k2) == 0):
#             Upload_edge_support_vector_ = k1
#             Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_), axis=0)
#
#             Upload_edge_label = Edge_label[i][new_support_plus[i][:1]]
#
#             new_support_plus[i], new_support_vector_plus[i] \
#                 = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
#
#         else:
#             Upload_edge_support_vector_ = np.concatenate((k1, k2), axis=0)
#             Upload_support_vector_[i] = np.concatenate((Upload_support_vector_[i], Upload_edge_support_vector_), axis=0)
#
#             Upload_edge_label = np.concatenate(
#                 (Edge_label[i][new_support_plus[i][:1]], Edge_label[i][new_support_minus[i][:1]]), axis=0)
#
#             new_support_plus[i], new_support_vector_plus[i] \
#                 = Ed.local_upload(new_support_plus[i], new_support_vector_plus[i])
#             new_support_minus[i], new_support_vector_minus[i] \
#                 = Ed.local_upload(new_support_minus[i], new_support_vector_minus[i])
#
#         Upload_label[i] = np.concatenate((Upload_label[i], Upload_edge_label), axis=0)
#
#     Collect_support_vector_ = np.concatenate((Upload_support_vector_[0], Upload_support_vector_[1]))
#     Collect_label = np.concatenate((Upload_label[0], Upload_label[1])).reshape((-1))
#
#     if (Edge_node_n > 2):
#         for j in range(2, Edge_node_n):
#             Collect_support_vector_ = np.concatenate((Collect_support_vector_, Upload_support_vector_[j]))
#             Collect_label = np.concatenate((Collect_label, Upload_label[j].reshape(-1)))
#
#     old_support_vectors_ = test.support_vectors_
#
#     test = svm.SVC(C=C, kernel='rbf', gamma=gamma)
#     test.fit(Collect_support_vector_, Collect_label)
#
#     new_support_vectors_ = test.support_vectors_
#     # SVM_plot(Collect_support_vector_[:, 0], Collect_support_vector_[:, 1], Collect_label, test)
#
#     if(SV_compare(old_support_vectors_, new_support_vectors_)):
#         break
#
# print(np.size(Collect_label, axis=0))
# print(test.n_support_)
