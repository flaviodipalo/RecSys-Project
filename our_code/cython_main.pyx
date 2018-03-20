from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
import numpy as np
import math
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
import time

cdef class CythonEpoch:

    cdef double[:] users
    cdef double[:] movies
    cdef double[:] ratings
    cdef int n_users
    cdef int n_movies


    cdef double vector_sum(self, double[:] vector):

            cdef double adder = 0
            cdef int i

            for i in range(len(vector)):
                adder += vector[i]
            return adder


    cdef double cython_product_dense(self, vector, vector1):

        cdef double result = 0
        cdef int i

        for i in range(len(vector)):
            result += vector[i]*vector1[i]

        return result



    cdef double cython_product_sparse(self, URM_without, vector):

            cdef double result = 0
            cdef int i = 0
            cdef int x
            cdef int j = 0
            cdef int[:] URM_without_indices = URM_without.indices
            cdef double[:] URM_without_data = URM_without.data

            for x in range(len(URM_without_data)):
                result += URM_without_data[x]*vector[URM_without_indices[x], 0]

            return result


    cdef double[:, :] cython_product_t_column(self, URM_without, S, t_column_indices):

            cdef double[:, :] prediction = np.zeros((self.n_users, 1))
            cdef int i
            cdef int j = 0

            for i in t_column_indices:
                URM_without_indptr = URM_without.indptr
                URM_without_indices = URM_without.indices[URM_without_indptr[i]:URM_without_indptr[i+1]]
                URM_without_data = URM_without.data[URM_without_indptr[i]:URM_without_indptr[i + 1]]
                for x in range(len(URM_without_data)):
                    prediction[i, j] += URM_without_data[x]*S[URM_without_indices[x], 0]
            return prediction

    cdef double linalg_cython(self, matrix):
        cdef int i, j
        cdef double counter = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                counter += matrix[i, j]**2

        return math.sqrt(counter)


    def __init__(self):

        data_reader = Movielens1MReader(0.8)

        URM_train = data_reader.URM_train
        URM_test = data_reader.URM_test

        self.users = data_reader.users
        self.movies = data_reader.movies
        self.ratings = data_reader.ratings
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]

        cdef int i = 0
        cdef int j = 1
        cdef double alpha = 1e-2
        cdef int gamma = 1
        cdef double beta = 1e-2
        cdef double[:,:] S = np.random.rand(self.n_movies,1)
        cdef int[:] URM_without_indptr, URM_without_indices, URM_without_data, t_column_indices
        cdef double[:] t_column_data
        cdef double [:, :] prediction, error
        cdef double[:, :] G
        cdef double gradient
        cdef double[:, :] prova_vector = np.zeros((self.n_movies, 1))

        for i in range(prova_vector.shape[0]):
            prova_vector[i, 0] = 5.0

        csc_URM_train = URM_train.tocsc()
        csc_URM_train_indptr = csc_URM_train.indptr
        t_column_indices = csc_URM_train.indices[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]
        t_column_data = csc_URM_train.data[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]

        #python passa le cose per riferimento, noi siamo interessati a copiarne i valori.
        URM_without = URM_train.copy()

        #TODO: è lento perché stiamo cambiando i valori di una matrice sparsa
        URM_without[:,j] = np.zeros((self.n_users,1))

        #these print are used for testing purposes.
        #print('prediction vector is:', cython_product(URM_without,S))
        #print('other vector is:', URM_without.dot(S))

        t_column = URM_train[:, j]

        #prediction = np.zeros((t_column.shape[0], 1))
        #error = np.zeros((t_column.shape[0], 1))
        prediction = np.zeros((self.n_users, 1))
        error = np.zeros((self.n_users, 1))
        #from not cython implementation

        #previous_error_function = np.linalg.norm(URM_without.dot(S)-t_column,2) +gamma*np.linalg.norm(S,2) +beta*np.linalg.norm(S)**2
        #current_error_function = np.linalg.norm(cython_product_t_column(URM_without, S, t_column_indices),2)+ gamma*np.linalg.norm(S,2)+beta*np.linalg.norm(S)**2
        #print(previous_error_function,error_function)

        # Needed for Adagrad
        G = np.zeros((self.n_movies, 1))
        eps = 1e-8

        start_time = time.time()
        #for l in range(10):
        while True:
            for i, e in enumerate(t_column_data):
                if e != 0:
                    #prediction[i, 0] = URM_without[i, :].dot(S)
                    prediction[i, 0] = self.cython_product_sparse(URM_without[i, :], S)
            print("the sum of the element is: ", self.vector_sum(prediction[:, 0]))

            for i in range(t_column_data.shape[0]):
                #error[i, 0] = prediction[i, 0] - t_column[i]
                error[i, 0] = prediction[i, 0] - t_column_data[i]
            print("the sum of the errors is: ", self.vector_sum(error[:, 0]))

            j = 1

            for i in range(self.n_movies):
                #gradient = (error[i, 0]*URM_without[j,i] + gamma + beta*S[i, 0])
                #gradient = (prova_vector[i, 0]*URM_without[j,i] + gamma + beta*S[i, 0])
                gradient = -self.cython_product_sparse(URM_without[:, j], error) + gamma + beta*S[i, 0]
                G[i, 0] += gradient**2
                S[i, 0] -= (alpha/math.sqrt(G[i, 0] + eps))*gradient

            #S -= (alpha * error * URM_without[j, :] - gamma*np.ones((self.n_movies,1)) - beta * S)
            error_function = self.linalg_cython(self.cython_product_t_column(URM_without, S, t_column_indices)) + gamma * self.linalg_cython(S) + beta * self.linalg_cython(S) ** 2
            #error_function1 = np.linalg.norm(self.cython_product_t_column(URM_without, S, t_column_indices),2) + gamma * np.linalg.norm(S, 2) + beta * np.linalg.norm(S) ** 2
            print(error_function)
        #print("The total time for %s iterations is %s seconds" %(l+1, time.time()-start_time))


    '''
    gradient_update = np.zeros((self.n_movies,1))
        while True:
            start = timeit.default_timer()
            for t in range (0,n_movies):
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
        S = np.random.rand(self.n_movies, 1)
        S[0, 0] = 0

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
                gradient_update[t] = np.linalg.norm(URM_train[t,j]-URM_train[t,:].dot(S),2)*-URM_train[j,t] +beta*np.abs(S[t, 0])
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




