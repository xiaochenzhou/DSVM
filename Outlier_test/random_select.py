from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import sklearn
import numpy as np

def random_select (global_num, Edge_data, Edge_label, Edge_support_vector):
    data, label = shuffle(Edge_data, Edge_label,random_state=0)
    # data, label = shuffle(data, label,random_state=0)

    Edge_num = np.size(Edge_label)
    interval = 5
    index = 0
    return_data = np.array ([Edge_data[index]]  )
    return_label = np.array ([Edge_label[index]] )
    index += interval
    # print (return_label)
    while (index < Edge_num):
        return_data = np.concatenate( (return_data, np.array ([data[index]]  )), axis = 0 )
        return_label = np.concatenate ( (return_label, np.array ([label[index]] )), axis = 0  )
        index += interval
    index = 0
    # print (return_data)
    print ("befor delete the return data size is ")
    print (np.size(return_data, axis = 0 ))
    for i in return_data:
        for k in Edge_support_vector:
            if ((i[0] == k[0]) & (i[1] == k[1]) & (i[2] == k[2]) & (i[3] == k[3]) & (i[4] == k[4]) ):
                return_data = np.delete(return_data, index, 0)
                return_label = np.delete(return_label, index, 0)
                index -= 1
                break
        index += 1
    print ("Edge node size is %s" %(np.size(Edge_label)))
    print ("this is the new data should be uploaded")
    # print (return_data)
    print (np.size(return_data, axis = 0))
    return return_data, return_label


def k_means_random_select (global_num, Edge_data, Edge_label, Edge_support_vector):
    data, label = shuffle(Edge_data, Edge_label,random_state=2)
    # data, label = shuffle(data, label,random_state=0)

    Edge_num = np.size(Edge_label)
    interval = 50
    k_node_num = (int) (np.size(Edge_label) / interval / 2)
    index = 0
    for i in Edge_data:
        if (Edge_label[index] == 1):
            Edge_data_plus = np.array( [ Edge_data[index] ]  )
            break
        index += 1
    index = 0
    for i in Edge_data:
        if (Edge_label[index] == -1):
            Edge_data_minus = np.array( [ Edge_data[index] ]  )
            break
        index += 1
    
    flag_plus = False
    flag_minus = False
    index = 0
    for i in Edge_data:
        if (Edge_label[index] == 1):
            if (flag_plus):
                Edge_data_plus = np.concatenate ( (Edge_data_plus, np.array ([ Edge_data[index] ])  ), axis = 0 )
            flag_plus = True
        if (Edge_label[index] == -1):
            if (flag_minus):
                Edge_data_minus = np.concatenate ( (Edge_data_minus, np.array ([ Edge_data[index] ])  ), axis = 0 )
            flag_minus = True

        index += 1
    # print (return_label)
    K_plus = KMeans(n_clusters = k_node_num, random_state = 1)
    K_plus.fit (Edge_data_plus)
    K_minus = KMeans(n_clusters = k_node_num, random_state = 1)
    K_minus.fit (Edge_data_minus)
    return_data = np.concatenate (  (K_plus.cluster_centers_, K_minus.cluster_centers_), axis = 0 )
    return_label = np.concatenate  (  (  (np.zeros(k_node_num) + 1), (np.zeros(k_node_num) - 1  )   ), axis = 0   )
    # print (return_data)
    index = 0
    print (np.size(return_data, axis = 0 ))
    for i in return_data:
        for k in Edge_support_vector:
            if ((i[0] == k[0]) & (i[1] == k[1]) & (i[2] == k[2]) & (i[3] == k[3]) & (i[4] == k[4]) ):
                return_data = np.delete(return_data, index, 0)
                return_label = np.delete(return_label, index, 0)
                index -= 1
                break
        index += 1
    print ("Edge node size is %s" %(np.size(Edge_label)))
    print ("this is the new data should be uploaded")
    # print (return_data)
    print (np.size(return_data, axis = 0))
    return return_data, return_label