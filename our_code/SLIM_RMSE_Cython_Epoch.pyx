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
#####
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

    cdef long[:] unique_movies
    cdef double[:] users, movies, ratings
    cdef int[:] all_items_indptr, all_items_indices
    cdef double[:] all_items_data
    cdef int[:] URM_indptr,  URM_indices
    cdef double[:] URM_data
    cdef double[:, :] S
    cdef double[:, :] G
    cdef int n_users, n_movies, i_gamma
    cdef double alpha
    cdef int i_iterations
    cdef double i_beta

    def __init__(self, unique_movies, URM_train, learning_rate, gamma, beta, iterations):

        self.unique_movies = unique_movies
        self.i_beta = beta
        self.i_iterations = iterations
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]
        self.URM_indices = URM_train.indices
        self.URM_data = URM_train.data
        self.URM_indptr = URM_train.indptr
        self.i_gamma = gamma
        self.alpha = learning_rate

        csc_URM_train = URM_train.tocsc()
        self.all_items_indptr = csc_URM_train.indptr
        self.all_items_indices = csc_URM_train.indices
        self.all_items_data = csc_URM_train.data
        self.G = np.zeros((self.n_movies, self.n_movies))


        self.S = np.random.rand(self.n_movies, self.n_movies)

        # Needed for Adagrad

    def epochIteration_Cython(self):
        cdef double [:] prediction = np.zeros(self.n_users)
        cdef double eps = 1e-8

        cdef int[:] URM_without_indptr, t_column_indices, item_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data

        cdef double gradient

        cdef double error_function
        cdef double partial_error

        cdef int counter
        cdef int time_counter = 0
        cdef int[:] user_indices
        cdef double[:] user_data

        cdef int user_index, target_user_index
        cdef int j
        cdef int i
        cdef int index
        cdef int n_iter
        cdef int t_index

        #for j in self.unique_movies:
        for j in range(0, self.n_movies):
            #if j%500 ==0:
            #    print("Column ", j)

            #t_column_indices = item_indices[item_indptr[j]:item_indptr[j+1]]
            #t_column_data = item_data[item_indptr[j]:item_indptr[j+1]]

            item_indices = self.all_items_indices[self.all_items_indptr[j]:self.all_items_indptr[j+1]]
            for n_iter in range(self.i_iterations):
                #if n_iter % 100 == 0:
                    #print("Iteration %d of column %d\n", n_iter, j)
                counter = 0
                for t_index in range(item_indices.shape[0]):
                    user_index = item_indices[t_index]
                    user_indices = self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]]
                    user_data = self.URM_data[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]]
                    partial_error = (cython_product_sparse(user_indices, user_data, self.S[:, j], j) - self.all_items_data[self.all_items_indptr[j]:self.all_items_indptr[j+1]][counter])

                    for index in range(user_indices.shape[0]):
                        target_user_index = user_indices[index]
                        if target_user_index != j:
                            gradient = partial_error*user_data[index]+ self.i_beta*self.S[target_user_index, j] + self.i_gamma
                            self.G[target_user_index, j] += gradient**2
                            self.S[target_user_index, j] -= (self.alpha/sqrt(self.G[target_user_index, j] + eps))*gradient

                        if self.S[target_user_index, j] < 0:
                            self.S[target_user_index, j] = 0
                    counter = counter + 1
            self.S[j, j] = 0

        #error_function = cython_norm(prediction_error(self.URM_indptr, self.URM_indices, self.URM_data, self.S[:, j], self.item_indices[self.item_indptr[j]:self.item_indptr[j+1]], self.item_data[self.item_indptr[j]:self.item_indptr[j+1]], j, prediction), 2)**2 + self.i_beta*cython_norm(self.S[:, j], 2)**2  + self.i_gamma*cython_norm(self.S[:, j], 1)


    def get_S(self):
        #self.S = np.random.rand(self.n_movies, self.n_movies)
        return np.asarray(self.S)