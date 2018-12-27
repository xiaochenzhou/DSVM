import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

def data_partition(train_data, train_label, E_node_n):

    data_index = np.arange(np.size(train_data, 0))

    data, label, index = shuffle(train_data, train_label, data_index, random_state = 0)

    data_part = [data for i in range(E_node_n)]
    label_part = [label for i in range(E_node_n)]
    index_part = [index for i in range(E_node_n)]

    local_data_n = int(np.size(train_label, 0) / E_node_n)

    for i in range(E_node_n):
        data_part[i] = data[i*local_data_n:(i+1)*local_data_n]
        label_part[i] = label[i*local_data_n:(i+1)*local_data_n]
        index_part[i] = index[i*local_data_n:(i+1)*local_data_n]

    return data_part, label_part, index_part

def kmeans_partition(train_data, train_label, E_node_n):
    train_data_plus = np.array([train_data[0]])
    train_data_minus = np.array([train_data[0]])
    train_label_plus = np.array([train_label[0]])
    train_label_minus = np.array([train_label[0]])
    index= 0
    f1 = True
    f2 = True
    for i in train_label:
        if (train_label[index] == 1):
            if (f1):
                train_data_plus = np.array([train_data[index]])
                train_label_plus = np.array([train_label[index]])
            else:
                train_data_plus = np.concatenate  (  (train_data_plus,np.array([train_data[index]]) ) , axis = 0   )
                train_label_plus = np.concatenate  (  (train_label_plus,np.array([train_label[index]]) ) , axis = 0   )
            f1 = False
        if (train_label[index] == -1):
            if (f2):
                train_data_minus = np.array([train_data[index]])
                train_label_minus = np.array([train_label[index]])
            else:
                train_data_minus = np.concatenate  (  (train_data_minus,np.array([train_data[index]]) ) , axis = 0   )
                train_label_minus = np.concatenate  (  (train_label_minus,np.array([train_label[index]]) ) , axis = 0   )
            f2 = False
        index += 1
    
    train_data = np.concatenate ( (train_data_plus, train_data_minus), axis  =0  )
    train_label = np.concatenate ( (train_label_plus, train_label_minus), axis = 0  )

    # print ("train_data.size is %i" %(np.size ( train_data, axis= 0 )) )
        
        

    half_data = int(np.size(train_label) / 2)

    k_means_plus = KMeans(n_clusters = E_node_n, random_state = 0).fit(train_data[:half_data])
    k_means_minus = KMeans(n_clusters=E_node_n, random_state=0).fit(train_data[half_data:])
    cluster_plus = k_means_plus.predict(train_data[:half_data])
    cluster_minus = k_means_minus.predict(train_data[half_data:])

    data_part = [train_data for i in range(E_node_n)]
    label_part = [train_label for i in range(E_node_n)]
    index_part = [train_label for i in range(E_node_n)]

    for i in range(E_node_n):
        index_plus = np.argwhere((cluster_plus == i)).reshape((-1))
        index_minus = np.argwhere((cluster_minus == i)).reshape((-1))
        data_plus = train_data[index_plus]
        data_minus = train_data[index_minus+half_data]
        label_plus = train_label[index_plus]
        label_minus = train_label[index_minus+half_data]

        index_part[i] = np.concatenate((index_plus, index_minus))
        data_part[i] = np.concatenate((data_plus, data_minus))
        label_part[i] = np.concatenate((label_plus, label_minus))

    return data_part, label_part, index_part

