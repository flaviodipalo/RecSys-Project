from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
import numpy as np
import math
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
import timeit


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
    n_movies = URM_train[:,:].shape[1]
    #S = np.random.rand(n_movies, 1)
    i = 0
    j = 1
    alpha = 1e-1
    gamma = 1
    beta = 1e-2

    print('nonzero element on the selcted column:', URM_train[:,j].nnz)



    #TODO: questa parte è per il cython
    j = 1

    S = np.random.rand(n_movies,1)
    csc_URM_train = URM_train.tocsc()
    csc_URM_train_indptr = csc_URM_train.indptr
    t_column_indices = csc_URM_train.indices[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]
    t_column_data = csc_URM_train.data[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]

    #python passa le cose per riferimento, noi siamo interessati a copiarne i valori.
    URM_without = URM_train.copy()
    #TODO: questo qui potrebbe essere lento
    URM_without[:,j] = np.zeros((URM_train.shape[0],1))
    print("DOPO: ", URM_train[:, 1])

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

    prediction = np.zeros((t_column.shape[0],1))
    error = np.zeros((t_column.shape[0],1))
    #from not cython implementation

    #previous_error_function = np.linalg.norm(URM_without.dot(S)-t_column,2) +gamma*np.linalg.norm(S,2) +beta*np.linalg.norm(S)**2
    error_function = np.linalg.norm(cython_product_t_column(URM_without, S, t_column_indices),2)+ gamma*np.linalg.norm(S,2)+beta*np.linalg.norm(S)**2
    #print(previous_error_function,error_function)

    # Needed for Adagrad
    G = np.zeros(np.size(S))
    eps = 1e-5

    while True:
        for i, e in enumerate(t_column):
            if e != 0:
                prediction[i] = URM_without[i, :].dot(S)
        print('the sum of the element is: ',prediction.sum())

        for i,j in enumerate(t_column):
            error[i] = prediction[i] - t_column[i]
        print('the sum of the errors is: ',error.sum())

        j = 1
        for i in range(n_movies):
            gradient = (error[i]*URM_without[j,i] + gamma + beta*S[i])
            G[i] += gradient**2
            S[i] -= (alpha/math.sqrt(G[i] + eps))*gradient

        #S -= (alpha * error * URM_without[j, :] - gamma*np.ones((n_movies,1)) - beta * S)
        error_function = np.linalg.norm(cython_product_t_column(URM_without, S, t_column_indices),2) + gamma * np.linalg.norm(S, 2) + beta * np.linalg.norm(S) ** 2
        print(error_function)

'''
gradient_update = np.zeros((n_movies,1))
    while True:
        start = timeit.default_timer()
        for t in range (0,S.shape[0]):
            gradient_update[t] = (URM_train[:,:].dot(S) - URM_train[:,j])*URM_train[j,t]
        stop = timeit.default_timer()
        print('gradient update took: ', stop-start)
        S = S + alpha*gradient_update
        new_evaluation = np.linalg.norm(URM_train[:, j] - URM_train.dot(S), 2)
        print('previous eval:',previous_evalutation,'new eval: ',new_evaluation)

    print(URM_train[:,1])
    #passiamo a calcolare la prediction ora.
    print(URM_train[:,:].shape[1])
    # we initialize the first colum of the S matrix (also called W matrix)
    S = np.random.rand(n_movies, 1)
    S[0] = 0

    #frobenius norm between the prediction and the value.
    #for the first step let's immagine we want to minimize this:
    i = 0
    j = 1
    alpha = 10^-12
    beta = 0.1

    print(URM_train)
    prediction = (URM_train).dot(S)
    print(prediction)
    previous_evalutation = np.linalg.norm(URM_train[:,j]-URM_train.dot(S),2)
    difference = URM_train[:, j] - URM_train.dot(S)
    print('first evaluation',previous_evalutation)
    print('difference vector',difference)
    #gradient deve essere pari lungo 3953, uno per ogni peso.
    #notes on gradient on ipad.

    gradient_update = np.zeros((3953,1))
    sum = 0
    t = 0
    j = 1
    backup_S = []
    while True:
        start = timeit.default_timer()
        #batch gradiente descent.
        #pick a random number between 0, and shape - 1
        #range
        #print(URM_train)
        #print(URM_train[t,:].shape)
        #print(S.shape)
        #print((URM_train[t,:].dot(S)).shape)

        for t in range (0,S.shape[0]):
            gradient_update[t] = np.linalg.norm(URM_train[t,j]-URM_train[t,:].dot(S),2)*-URM_train[j,t] +beta*np.abs(S[t])
        stop = timeit.default_timer()
        print('gradient update took: ', stop-start)
        print(np.sum(alpha*gradient_update))
        S = S + alpha * gradient_update
        backup_S.append(np.sum(S))
        print(backup_S)
        new_evaluation = np.linalg.norm(URM_train[:, j]-URM_train.dot(S),2)
        print('previous evalutation: ',previous_evalutation,'new evaluation: ', new_evaluation)
        previous_evalutation = new_evaluation
    #print(gradient_update)

    #la stima della stessa colonna imparata è
#    recommender_list = []
    #recommender_list.append(SLIM_BPR_Cython(URM_train, sparse_weights=False))
#    recommender_list.append(SLIM_RMSE(URM_train))
    rec_object = SLIM_RMSE(URM_train)
    #add cython compiling
    #rec_object.SLIM_RMSE_epoch(URM_train)
    rec_object.SLIM_RMSE_epoch(URM_train, users, movies, ratings, users_by_item, items_by_item, ratings_by_item) #prova

#inizializziamo la W random con la diagonale a 0.
'''
'''
    for recommender in recommender_list:

        print("Algorithm: {}".format(recommender.__class__))

        recommender.fit()

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        print("Algorithm: {}, results: {}".format(recommender.__class__, results_run))
'''
