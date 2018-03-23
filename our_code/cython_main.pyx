from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
import numpy as np
#import math
from libc.math cimport sqrt
import random
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
#python setup.py build_ext --inplace

import time

cdef class CythonEpoch:

    cdef double[:] users
    cdef double[:] movies
    cdef double[:] ratings
    cdef int n_users
    cdef int n_movies

    #TODO: meglio così o numpy ?

    cdef double vector_sum(self, double[:] vector):

            cdef double adder = 0
            cdef int i

            for i in range(len(vector)):
                adder += vector[i]
            return adder

    #TODO: dichiarare i tipi dei parametri
    #cdef double cython_product_sparse(self, URM_without, double[:] vector):
    cdef double cython_product_sparse(self, int[:] URM_without_indices, double[:] URM_without_data, double[:] vector):

            cdef double result = 0
            cdef int i = 0
            cdef int x
            cdef int j = 0


            for x in range(len(URM_without_data)):
                result += URM_without_data[x]*vector[URM_without_indices[x]]

            return result

    cdef double[:, :] cython_product_t_column(self, URM_without, S, t_column_indices):

            cdef double[:, :] prediction = np.zeros((self.n_users, 1))
            cdef int i
            #TODO: questo è impostato per essere solo su una colonna
            cdef int j = 0

            for i in t_column_indices:
                URM_without_indptr = URM_without.indptr
                URM_without_indices = URM_without.indices[URM_without_indptr[i]:URM_without_indptr[i+1]]
                URM_without_data = URM_without.data[URM_without_indptr[i]:URM_without_indptr[i + 1]]
                for x in range(len(URM_without_data)):
                    prediction[i, j] += URM_without_data[x]*S[URM_without_indices[x]]
            return prediction

    cdef double cython_norm(self, matrix, option):
        cdef int i, j
        cdef double counter = 0

        if option == 2:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    counter += matrix[i, j]**2
        elif option == 1:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    counter += matrix[i, j]

        return sqrt(counter)

    def __init__(self):

        data_reader = Movielens1MReader(0.8)

        URM_train = data_reader.URM_train
        URM_test = data_reader.URM_test

        self.users = data_reader.users
        self.movies = data_reader.movies
        self.ratings = data_reader.ratings
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]

        cdef int user = 0
        #TODO: j è fisso per ora siccome abbiamo considerato una sola colonna
        cdef int j = 1
        cdef double alpha = 1e-1
        cdef int gamma = 5
        cdef double beta = 1e-2
        cdef int iterations = 1000
        cdef int threshold = 5
        cdef double[:] S = np.random.rand(self.n_movies)
        cdef int[:] URM_without_indptr, t_column_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data
        cdef double [:, :] prediction, error
        cdef double[:] G
        cdef double gradient
        cdef double error_function
        cdef double [:, :] max_arg_s = np.zeros((iterations, 1))
        cdef double partial_error
        cdef double[:] URM_data
        cdef int[:] URM_indices
        cdef int[:] URM_indptr
        cdef int counter
        cdef int time_counter = 0
        cdef int[:] URM_vector_indices
        cdef double[:] URM_vector_data

        csc_URM_train = URM_train.tocsc()
        csc_URM_train_indptr = csc_URM_train.indptr
        t_column_indices = csc_URM_train.indices[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]
        t_column_data = csc_URM_train.data[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]

        #python passa le cose per riferimento, noi siamo interessati a copiarne i valori.
        URM_without = URM_train.copy()

        #TODO: è lento perché stiamo cambiando i valori di una matrice sparsa
        URM_without[:,j] = np.zeros((self.n_users,1))

        t_column = URM_train[:, j]
        prediction = np.zeros((self.n_users, 1))
        error = np.zeros((self.n_users, 1))

        #previous_error_function = np.linalg.norm(URM_without.dot(S)-t_column,2) +gamma*np.linalg.norm(S,2) +beta*np.linalg.norm(S)**2
        #current_error_function = np.linalg.norm(cython_product_t_column(URM_without, S, t_column_indices),2)+ gamma*np.linalg.norm(S,2)+beta*np.linalg.norm(S)**2
        #print(previous_error_function,error_function)

        # Needed for Adagrad
        G = np.zeros((self.n_movies))
        eps = 1e-8

        URM_indices = URM_without.indices
        URM_data = URM_without.data
        URM_indptr = URM_without.indptr

        for n_iter in range(iterations):
            print("Iteration #%s" %(n_iter))
            '''
            if n_iter%100 == 0:
                for i, e in enumerate(t_column_data):
                    if e != 0:
                        URM_vector_indices = URM_without[i, :].indices
                        URM_vector_data = URM_without[i, :].data
                        prediction[i, 0] = self.cython_product_sparse(URM_vector_indices,URM_vector_data, S)

                print("the sum of the element is: ", self.vector_sum(prediction[:, 0]))

                for i in range(t_column_data.shape[0]):
                    error[i, 0] = prediction[i, 0] - t_column_data[i]
                print("the sum of the errors is: ", self.vector_sum(error[:, 0]))
            '''

            j = 1
            counter = 0
            start_time = time.time()
            #TODO: change indices.
            for user in t_column_indices:
                time_counter += 1
                if time_counter % 1000 == 0:
                    print("Time for 1000 iterations: ", time_counter/(time.time() - start_time))
                ##no sgd
                #print('sono arrivato a:',i)

                '''
                for gradient_index in range (self.n_movies):
                    print('i, gradient:',i,gradient)
                    gradient = (self.cython_product_sparse(URM_without[i, :], S) - t_column[i, 0])*URM_without[i, gradient_index] + gamma + beta*S[gradient_index, 0]
                    G[gradient_index, 0] += gradient**2
                    S[gradient_index, 0] -= (alpha/math.sqrt(G[i, 0] + eps))*gradient
                    if S[gradient_index, 0] < 0:
                        S[gradient_index, 0] = 0
                '''
                URM_vector_indices = URM_indices[URM_indptr[user]:URM_indptr[user+1]]
                URM_vector_data = URM_data[URM_indptr[user]:URM_indptr[user+1]]
                partial_error = (self.cython_product_sparse(URM_vector_indices, URM_vector_data, S) - t_column_data[counter])
                #for gradient_index in range (self.n_movies):
                for index in range(len(URM_vector_indices)):

                    #gradient_index = random.randrange(0, self.n_movies-1, 1)
                    #print('i, gradient:',i,gradient)
                    gradient = partial_error*URM_vector_data[index] + gamma + beta*S[URM_vector_indices[index]]
                    G[URM_vector_indices[index]] += gradient**2
                    S[URM_vector_indices[index]] -= (alpha/sqrt(G[URM_vector_indices[index]] + eps))*gradient
                    if S[URM_vector_indices[index]] < 0:
                        S[URM_vector_indices[index]] = 0
                counter = counter + 1

            error_function = np.linalg.norm(self.cython_product_t_column(URM_without, S, t_column_indices), 2)**2 + beta*np.linalg.norm(S, 2)**2  + gamma*np.linalg.norm(S, 1)

            if error_function < threshold:
                break
            print('error function is: ',error_function)

        print("The total time for %s iterations is %s seconds" %(n_iter+1, time.time()-start_time))

        #print prediction for all the values different from zero.
        '''
        for i in range(self.n_users):
            if (URM_train[i, j] != 0):
                print("Real: %s    predicted: %s" %(URM_train[i, 1], self.cython_product_sparse(URM_without[i, :], S)))
        '''