from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
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
#TODO: ristrutturare il codice ed aggiungere i metodi  fit etc
#TODO: valutare il codice secondo le metriche sfruttando il codice di MFD.
#TODO: parallelizzare

#TODO: cambiare il gradient e provare con la nostra alternativa

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double cython_product_sparse(int[:] URM_indices, double[:] URM_data, double[:] S_column, int column_index_with_zero) nogil:

        cdef double result = 0
        cdef int x

        for x in range(len(URM_data)):
            if URM_indices[x] != column_index_with_zero:
                result += URM_data[x]*S_column[URM_indices[x]]

        return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:] prediction_error(int[:] URM_indptr, int[:] URM_indices, double[:] URM_data, double[:] S, int[:] t_column_indices, double[:] t_column_data, int column_index_with_zero, double[:] prediction) nogil:

        #cdef double[:] prediction = np.zeros(len(t_column_indices))
        cdef int x, user, index, i
        cdef int[:] user_indices
        cdef double[:] user_data

        for index in range(len(t_column_indices)):
            user = t_column_indices[index]
            user_indices = URM_indices[URM_indptr[user]:URM_indptr[user + 1]]
            user_data = URM_data[URM_indptr[user]:URM_indptr[user + 1]]

            prediction[index] = 0
            for x in range(len(user_data)):
                if user_indices[x] != column_index_with_zero:
                    prediction[index] += user_data[x]*S[user_indices[x]]
            prediction[index] = t_column_data[index] - prediction[index]

        return prediction

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double cython_norm(double[:] vector, int option) nogil:

    cdef int i
    cdef double counter = 0

    if option == 2:
        for i in range(len(vector)):
            counter += vector[i]**2
        counter = sqrt(counter)
    elif option == 1:
        for i in range(len(vector)):
            counter += vector[i]

    return counter


cdef class CythonEpoch:

    cdef double[:] users
    cdef double[:] movies

    cdef double[:] ratings
    cdef int n_users
    cdef int n_movies


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __init__(self):
        data_reader = Movielens1MReader(0.8)
        URM_train = data_reader.URM_train
        URM_test = data_reader.URM_test
        self.users = data_reader.users
        self.movies = data_reader.movies
        self.ratings = data_reader.ratings
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]

        cdef int user_index
        cdef int j
        cdef int i
        cdef int index
        cdef int n_iter
        cdef int t_index
        cdef double alpha = 1e-1
        cdef int gamma = 5
        cdef double beta = 1e-2
        cdef int iterations = 500
        cdef int threshold = 5
        cdef double[:, :] S = np.random.rand(self.n_movies, self.n_movies)
        cdef int[:] URM_without_indptr, t_column_indices, item_indptr, item_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data
        cdef double [:] prediction = np.zeros(self.n_users)
        cdef double[:, :] G
        cdef double gradient

        cdef double error_function
        cdef double partial_error
        cdef double[:] URM_data, item_data
        cdef int[:] URM_indices
        cdef int[:] URM_indptr
        cdef int counter
        cdef int time_counter = 0
        cdef int[:] URM_vector_indices
        cdef double[:] URM_vector_data



        #python passa le cose per riferimento, noi siamo interessati a copiarne i valori.
        #TODO: è lento perché stiamo cambiando i valori di una matrice sparsa

        # Needed for Adagrad
        G = np.zeros((self.n_movies, self.n_movies))
        eps = 1e-8

        URM_indices = URM_train.indices
        URM_data = URM_train.data
        URM_indptr = URM_train.indptr

        csc_URM_train = URM_train.tocsc()
        item_indptr = csc_URM_train.indptr
        item_indices = csc_URM_train.indices
        item_data = csc_URM_train.data

        with nogil, parallel():
            for j in prange(1, self.n_movies):
                printf("Column %d\n", j)
                #t_column_indices = item_indices[item_indptr[j]:item_indptr[j+1]]
                #t_column_data = item_data[item_indptr[j]:item_indptr[j+1]]

                for n_iter in range(iterations):
                    if n_iter % 100 == 0:
                        printf("Iteration #%d of column #%d\n", n_iter, j)

                    counter = 0
                    for t_index in range(len(item_indices[item_indptr[j]:item_indptr[j+1]])):
                        user_index = item_indices[item_indptr[j]:item_indptr[j+1]][t_index]
                        #URM_vector_indices = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]]
                        #URM_vector_data = URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]]
                        partial_error = (cython_product_sparse(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]], URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]], S[:, j], j) - item_data[item_indptr[j]:item_indptr[j+1]][counter])

                        for index in range(len(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]])):
                            if URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index] != j:
                                gradient = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index]+ beta*S[URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index], j] + gamma
                                G[URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index], j] += gradient**2
                                S[URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index], j] -= (alpha/sqrt(G[URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index], j] + eps))*gradient
                            if S[URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index], j] < 0:
                                S[URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index], j] = 0
                        counter = counter + 1

                    error_function = cython_norm(prediction_error(URM_indptr, URM_indices, URM_data, S[:, j], item_indices[item_indptr[j]:item_indptr[j+1]], item_data[item_indptr[j]:item_indptr[j+1]], j, prediction), 2)**2 + beta*cython_norm(S[:, j], 2)**2  + gamma*cython_norm(S[:, j], 1)

                    #if error_function < threshold:
                     #   break
                    #if n_iter % 100 == 0:
                     #   print("error function is: ", error_function)

                #print prediction for all the values different from zero.
                #for i in range(self.n_users):
                #    if (URM_train[i, j] != 0):
                #        print("Real: %s    predicted: %s" %(URM_train[i, j], cython_product_sparse(URM_indices[URM_indptr[i]:URM_indptr[i+1]],URM_data[URM_indptr[i]:URM_indptr[i+1]], S[:, j], j)))

