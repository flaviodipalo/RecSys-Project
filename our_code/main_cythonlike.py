from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
import numpy as np
import math
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
import time


if __name__ == '__main__':
    data_reader = Movielens1MReader(0.8)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test

    users = data_reader.users
    movies = data_reader.movies
    ratings = data_reader.ratings

    '''
    users_by_item = data_reader.users_by_item
    items_by_item = data_reader.items_by_item
    ratings_by_item = data_reader.ratings_by_item
    '''
    n_users = URM_train[:,:].shape[0]
    n_movies = URM_train[:,:].shape[1]
    #S = np.random.rand(n_movies, 1)
    i = 0
    j = 1
    alpha = 1e-1
    gamma = 20
    beta = 1e-2
    stop_when_increase = False

    print('nonzero element on the selcted column:', URM_train[:,j].nnz)

    #TODO: questa parte è per il cython
    j = 1

    S = np.random.rand(n_movies,1)
    S[0] = 0
    S_temp = S.copy()
    csc_URM_train = URM_train.tocsc()
    csc_URM_train_indptr = csc_URM_train.indptr
    t_column_indices = csc_URM_train.indices[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]
    t_column_data = csc_URM_train.data[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]

    #python passa le cose per riferimento, noi siamo interessati a copiarne i valori.
    URM_without = URM_train.copy()
    #TODO: questo qui potrebbe essere lento
    URM_without[:,j] = np.zeros((URM_train.shape[0],1))

    def cython_product(URM_without,S):
        prediction = np.zeros((URM_without.shape[0], 1))
        for i in range(URM_without.shape[0]):
            URM_without_indptr = URM_without.indptr
            URM_without_indices = URM_without.indices[URM_without_indptr[i]:URM_without_indptr[i+1]]
            URM_without_data = URM_without.data[URM_without_indptr[i]:URM_without_indptr[i + 1]]
            adder = 0
            for x in range(len(URM_without_data)):
                adder = adder + URM_without_data[x]*S[URM_without_indices[x]]
            prediction[i] = adder
        return prediction

    #these print are used for testing purposes.
    #print('prediction vector is:',cython_product(URM_without,S))
    #print('other vector is:',URM_without.dot(S))

    def cython_product_t_column(URM_without,S,t_column_indices):
        prediction = np.zeros((URM_without.shape[0], 1))
        for i in t_column_indices:
            URM_without_indptr = URM_without.indptr
            URM_without_indices = URM_without.indices[URM_without_indptr[i]:URM_without_indptr[i+1]]
            URM_without_data = URM_without.data[URM_without_indptr[i]:URM_without_indptr[i + 1]]
            adder = 0
            for x in range(len(URM_without_data)):
                adder = adder + URM_without_data[x]*S[URM_without_indices[x]]
            prediction[i] = adder
        return prediction
    #t_column = URM_train[:,j]
    #pred = cython_product_t_column(URM_without, S, t_column_indices)
    #prova = t_column - pred
    #print('prediction vector is:', prova.sum())

    t_column = URM_train[:,j]

    prediction = np.zeros((n_users,1))
    error = np.zeros((n_users,1))
    #from not cython implementation

    #previous_error_function = np.linalg.norm(URM_without.dot(S)-t_column,2) +gamma*np.linalg.norm(S,2) +beta*np.linalg.norm(S)**2
    error_function = np.linalg.norm(cython_product_t_column(URM_without, S, t_column_indices),2)+ gamma*np.linalg.norm(S,2)+beta*np.linalg.norm(S)**2
    #print(previous_error_function,error_function)

    max_arg_s = np.zeros((100, 1))
    # Needed for Adagrad
    G = np.zeros(np.size(S))
    eps = 1e-8
    start_time = time.time()
    for n_iter in range(100):
        error_function = 0.5 * (np.linalg.norm(URM_without.dot(S) - t_column, 2) ** 2) + beta / 2 * np.linalg.norm(S,
                                                                                                                   2) ** 2 + gamma * np.linalg.norm(
            S, 1)
        for i, e in enumerate(t_column):
            if e != 0:
                prediction[i] = URM_without[i, :].dot(S)
        print('the sum of the element is: ',prediction.sum())

        for i,j in enumerate(t_column):
            error[i] = prediction[i] - t_column[i]
        print('the sum of the errors is: ',error.sum())

        j = 1
        for i in range(n_movies):
            gradient = ((URM_without[n_iter, :].dot(S) - URM_train[n_iter, i]) * URM_train[n_iter, i] + beta * S[i] + gamma)
            G[i] += gradient**2
            S_temp[i] = S[i] - (alpha/math.sqrt(G[i] + eps))*gradient
            if S_temp[i] < 0:
                S_temp[i] = 0
        S_temp[0] = 0
        #S -= (alpha * error * URM_without[j, :] - gamma*np.ones((n_movies,1)) - beta * S)
        new_error_function = 0.5*(np.linalg.norm(URM_without.dot(S_temp) - t_column,2)**2) + beta/2 * np.linalg.norm(S_temp, 2)**2 + gamma* np.linalg.norm(S_temp, 1)
        if stop_when_increase:
            if new_error_function < error_function:
                S = S_temp.copy()
                max_arg_s[n_iter] = np.max(S)
                print("The max weight of S is %s" % (max_arg_s[n_iter]))
            else:
                break
        else:
            S = S_temp.copy()
            max_arg_s[n_iter] = np.max(S)
            print("The max weight of S is %s" %(max_arg_s[n_iter]))
        print("The total error is %s " %(new_error_function))
    print("%s seconds for %s iterations" % (time.time() - start_time, n_iter + 1))
    print(S)
    for i in range(n_users):
        if (URM_train[i, 1] != 0):
            print("Real: %s    predicted: %s" % (URM_train[i, 1], URM_train[i, :].dot(S)))


