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

def k_means_random_partition (train_data, train_label, E_node_n, k_centre_number, the_node_should_have_same_num_sample = False):

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

    t_size = np.size(train_label)

    half_data = int(t_size / 2)

    k_centre_number

    k_means_plus = KMeans(n_clusters = k_centre_number, random_state = 0).fit(train_data[:half_data])
    k_means_minus = KMeans(n_clusters=k_centre_number, random_state=0).fit(train_data[half_data:])
    cluster_plus = k_means_plus.predict(train_data[:half_data])
    cluster_minus = k_means_minus.predict(train_data[half_data:])

    data_t_p = [train_data for i in range(k_centre_number)]
    label_t_p = [train_label for i in range(k_centre_number)]
    index_t_p = [train_label for i in range(k_centre_number)]
    data_t_m = [train_data for i in range(k_centre_number)]
    label_t_m = [train_label for i in range(k_centre_number)]
    index_t_m = [train_label for i in range(k_centre_number)]

    for i in range(k_centre_number):
        index_plus = np.argwhere((cluster_plus == i)).reshape((-1))
        index_minus = np.argwhere((cluster_minus == i)).reshape((-1))
        data_plus = train_data[index_plus]
        data_minus = train_data[index_minus+half_data]
        label_plus = train_label[index_plus]
        label_minus = train_label[index_minus+half_data]

        index_t_p[i] = index_plus
        data_t_p[i] = data_plus
        label_t_p[i] = label_plus
        index_t_m[i] = index_minus
        data_t_m[i] = data_minus
        label_t_m[i] = label_minus


    data_part = [train_data for i in range(E_node_n)]
    label_part = [train_label for i in range(E_node_n)]
    index_part = [train_label for i in range(E_node_n)]


    rng = np.random.RandomState(1)

    for i in range(k_centre_number):
        random_select_coef_p = rng.rand(E_node_n)
        random_select_coef_m = rng.rand(E_node_n)
        t_p_num = np.size (index_t_p[i])
        t_m_num = np.size (index_t_m[i])

        p_sum = np.sum(random_select_coef_p)
        random_select_coef_p = random_select_coef_p / p_sum * t_p_num
        random_select_coef_p[E_node_n-1] = t_p_num

        m_sum = np.sum(random_select_coef_m)
        random_select_coef_m = random_select_coef_m / m_sum * t_m_num
        random_select_coef_m[E_node_n-1] = t_m_num
        
        for j in range(E_node_n):
            if (j >= 1):
                random_select_coef_m[j] = random_select_coef_m[j-1] + random_select_coef_m[j]
                random_select_coef_p[j] = random_select_coef_p[j-1] + random_select_coef_p[j]
        
        for j in range(E_node_n):
            past_p = 0
            past_m = 0
            if (j >= 1):
                past_p = random_select_coef_p[j-1]
                past_m = random_select_coef_m[j-1]
            # a = random_select_coef_m[j]
            # print (a)
            # print (index_t_m[i])
            in_m = index_t_m[i][ int(past_m): int(random_select_coef_m[j]) ]
            in_p = index_t_p[i][ int(past_p): int(random_select_coef_p[j]) ]
            da_m = data_t_m[i][ int(past_m): int(random_select_coef_m[j]) ]
            da_p = data_t_p[i][ int (past_p): int(random_select_coef_p[j]) ]
            la_m = label_t_m[i][ int(past_m): int(random_select_coef_m[j]) ]
            la_p = label_t_p[i][ int(past_p): int(random_select_coef_p[j]) ]

            if (i == 0):
                data_part[j] = np.concatenate (  (da_m, da_p), axis = 0)
                label_part[j] = np.concatenate (  (la_m, la_p), axis= 0 )
                index_part[j] = np.concatenate (  (in_m, in_p), axis = 0 )
            else :
                data_part[j] = np.concatenate (  (data_part[j], da_m, da_p), axis = 0)
                label_part[j] = np.concatenate (  (label_part[j], la_m, la_p), axis= 0 )
                index_part[j] = np.concatenate (  (index_part[j], in_m, in_p), axis = 0 )


    if (the_node_should_have_same_num_sample):
        for i in range(E_node_n):
            data_part, label_part, index_part = shuffle(data_part, label_part, index_part , random_state = 0)
        over_500 = []
        for i in range(E_node_n):
            if (np.size (label_part[i]) > 500  ):
                over_500.append(i)
        for i in over_500:
            while (np.size (label_part[i]) > 500):
                for j in range(E_node_n):
                    if (np.size (label_part[j]) >= 500):
                        continue
                    if (np.size (label_part[i]) <= 500):
                        break
                    label_t = np.copy(label_part[i][:1])
                    data_t = np.copy(data_part[i][:1])
                    index_t = np.copy(index_part[i][:1])

                    index_part[j] = np.concatenate ( (index_part[j], index_t), axis = 0 )
                    data_part[j] = np.concatenate ( (data_part[j], data_t), axis = 0 )
                    label_part[j] = np.concatenate ( (label_part[j], label_t), axis = 0 )

                    index_part[i] = np.delete(index_part[i], 0, 0)
                    label_part[i] = np.delete(label_part[i], 0, 0)
                    data_part[i] = np.delete(data_part[i], 0, 0)
                    


    return data_part, label_part, index_part
