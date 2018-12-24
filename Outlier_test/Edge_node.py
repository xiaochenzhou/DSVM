import numpy as np
from sklearn import svm
from sklearn.utils import shuffle

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

def local_train(train_data, train_label, C, gamma, kernel='rbf'):
    local_model = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    local_model.fit(train_data, train_label)

    return local_model

def local_support(local_model):
    support_ = local_model.support_
    support_vectors_ = local_model.support_vectors_
    n_support_ = local_model.n_support_

    return support_, support_vectors_, n_support_

def upload_sort(support_: object, support_vectors_: object, Distance: object) -> object:
    order = np.argsort(Distance)

    upload_support_vectors_ = support_vectors_[order]
    upload_support_ = support_[order]

    return upload_support_, upload_support_vectors_

def local_upload(support_, support_vectors_,):
    update_support_ = support_[1:]
    update_support_vectors_ = support_vectors_[1:,:]

    return update_support_, update_support_vectors_

def local_upload_outlier(support_, support_vectors_, outlier_n):
    update_support_ = support_[outlier_n:]
    update_support_vectors_ = support_vectors_[outlier_n:, :]

    return update_support_, update_support_vectors_

def data_mix(local_data, local_label, received_data, received_label):
    for i in range(np.size(received_data, axis=0)):
        if (not array_compare(received_data[i], local_data)):
            local_data = np.concatenate((local_data, [received_data[i]]), axis=0)
            local_label = np.concatenate((local_label, [received_label[i]]), axis=0)

    return local_data, local_label


