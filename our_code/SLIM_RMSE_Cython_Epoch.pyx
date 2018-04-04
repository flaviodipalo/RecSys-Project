from data.movielens_1m.Movielens1MReader import Movielens1MReader
from cython.parallel import parallel,  prange
cimport cython
from libc.stdio cimport printf


import numpy as np

from libc.math cimport sqrt
import random
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
#python setup.py build_ext --inplace
import time
import timeit
#TODO: portare le funzioni di prodotto fra matrici fuori dalla classe, idealmente in un nuovo file.

#TODO: valutare il codice secondo le metriche sfruttando il codice di MFD.
#TODO: parallelizzare

#TODO: cambiare il gradient e provare con la nostra alternativa


cdef double cython_product_sparse(int[:] URM_indices, double[:] URM_data, double[:] S_column, int column_index_with_zero):

        cdef double result = 0
        cdef int x

        for x in range(URM_data.shape[0]):
            if URM_indices[x] != column_index_with_zero:
                result += URM_data[x]*S_column[URM_indices[x]]

        return result


cdef double[:] prediction_error(int[:] URM_indptr, int[:] URM_indices, double[:] URM_data, double[:] S, int[:] t_column_indices, double[:] t_column_data, int column_index_with_zero, double[:] prediction):

        #cdef double[:] prediction = np.zeros(len(t_column_indices))
        cdef int x, user, index, i
        cdef int[:] user_indices
        cdef double[:] user_data

        for index in range(t_column_indices.shape[0]):
            user = t_column_indices[index]
            user_indices = URM_indices[URM_indptr[user]:URM_indptr[user + 1]]
            user_data = URM_data[URM_indptr[user]:URM_indptr[user + 1]]

            prediction[index] = 0
            for x in range(user_data.shape[0]):
                if user_indices[x] != column_index_with_zero:
                    prediction[index] += user_data[x]*S[user_indices[x]]
            prediction[index] = t_column_data[index] - prediction[index]

        return prediction

cdef double cython_norm(double[:] vector, int option):

    cdef int i
    cdef double counter = 0

    if option == 2:
        for i in range(vector.shape[0]):
            counter += vector[i]**2
        counter = sqrt(counter)
    elif option == 1:
        for i in range(vector.shape[0]):
            counter += vector[i]

    return counter


cdef class SLIM_RMSE_Cython_Epoch:

    cdef double[:] users, movies, ratings
    cdef int[:] item_indptr, item_indices
    cdef double[:] item_data
    cdef int[:] URM_indptr,  URM_indices
    cdef double[:] URM_data
    cdef double[:, :] S
    cdef int n_users, n_movies, i_gamma
    cdef double alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)

    def __init__(self, URM_train, learning_rate, gamma, beta, iterations):
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]
        self.URM_indices = URM_train.indices
        self.URM_data = URM_train.data
        self.URM_indptr = URM_train.indptr
        self.i_gamma = gamma
        self.alpha = learning_rate

        csc_URM_train = URM_train.tocsc()
        self.item_indptr = csc_URM_train.indptr
        self.item_indices = csc_URM_train.indices
        self.item_data = csc_URM_train.data
        cdef int[:] item_indptr
        cdef int[:] item_indices
        cdef double[:] item_data

        cdef int[:] URM_indptr
        cdef int[:] URM_indices
        cdef double[:] URM_data

        cdef int user_index
        cdef int j
        cdef int i
        cdef int index
        cdef int n_iter
        cdef int t_index



        cdef double i_beta = beta
        cdef int i_iterations = iterations
        cdef double eps = 1e-8

        self.S = np.random.rand(self.n_movies, self.n_movies)
        cdef int[:] URM_without_indptr, t_column_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data
        cdef double [:] prediction = np.zeros(self.n_users)
        cdef double[:, :] G
        cdef double gradient

        cdef double error_function
        cdef double partial_error

        cdef int counter
        cdef int time_counter = 0
        cdef int[:] URM_vector_indices
        cdef double[:] URM_vector_data

        # Needed for Adagrad
        G = np.zeros((self.n_movies, self.n_movies))

        item_indices = self.item_indices
        item_indptr = self.item_indptr
        item_data = self.item_data

        URM_indices = self.URM_indices
        URM_indptr = self.URM_indptr
        URM_data = self.URM_data

    def epochIteration_Cython(self):
        #TODO: aggiusta questo epochIteration per farlo andare.
        for j in range(1, self.n_movies):
            print("Column %d\n", j)

            #t_column_indices = item_indices[item_indptr[j]:item_indptr[j+1]]
            #t_column_data = item_data[item_indptr[j]:item_indptr[j+1]]

            for n_iter in range(self.i_iterations):
                if n_iter % 100 == 0:
                    print("Iteration #%d of column #%d\n", n_iter, j)

                counter = 0
                for t_index in range(self.item_indices[self.item_indptr[j]:self.item_indptr[j+1]].shape[0]):
                    user_index = self.item_indices[self.item_indptr[j]:self.item_indptr[j+1]][t_index]
                    #URM_vector_indices = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]]
                    #URM_vector_data = URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]]
                    partial_error = (cython_product_sparse(self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]], self.URM_data[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]], self.S[:, j], j) - self.item_data[self.item_indptr[j]:self.item_indptr[j+1]][counter])

                    for index in range(self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]].shape[0]):
                        if self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index] != j:
                            gradient = partial_error*self.URM_data[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index]+ self.i_beta*self.S[self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index], j] + self.i_gamma
                            self.G[self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index], j] += gradient**2
                            self.S[self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index], j] -= (self.alpha/sqrt(self.G[self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index], j] + self.eps))*gradient
                        if self.S[self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index], j] < 0:
                            self.S[self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]][index], j] = 0
                    counter = counter + 1

                error_function = cython_norm(prediction_error(self.URM_indptr, self.URM_indices, self.URM_data, self.S[:, j], self.item_indices[self.item_indptr[j]:self.item_indptr[j+1]], self.item_data[self.item_indptr[j]:self.item_indptr[j+1]], j, self.prediction), 2)**2 + self.i_beta*cython_norm(self.S[:, j], 2)**2  + self.i_gamma*cython_norm(self.S[:, j], 1)


    def get_S(self):
        return self.S

