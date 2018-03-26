from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE

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


cdef double vector_sum( double[:] vector):

        cdef double adder = 0
        cdef int i

        for i in range(len(vector)):
            adder += vector[i]
        return adder

cdef double cython_product_sparse( int[:] URM_without_indices, double[:] URM_without_data, double[:] vector):

        cdef double result = 0
        cdef int i = 0
        cdef int x
        cdef int j = 0


        for x in range(len(URM_without_data)):
            result += URM_without_data[x]*vector[URM_without_indices[x]]

        return result


cdef double[:] cython_product_t_column( URM_without, double[:] S, t_column_indices):

        cdef double[:] prediction = np.zeros(URM_without.shape[0])
        cdef int x, user, index, i
        cdef int[:] URM_without_indptr, URM_without_indices
        cdef double[:] URM_without_data

        for index in range(len(t_column_indices)):
            user = t_column_indices[index]
            URM_without_indptr = URM_without.indptr
            URM_without_indices = URM_without.indices[URM_without_indptr[user]:URM_without_indptr[user+1]]
            URM_without_data = URM_without.data[URM_without_indptr[user]:URM_without_indptr[user + 1]]
            for x in range(len(URM_without_data)):
                prediction[user] += URM_without_data[x]*S[URM_without_indices[x]]
        return prediction

cdef double cython_norm( matrix, option):
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



cdef class CythonEpoch:

    cdef double[:] users
    cdef double[:] movies
    cdef double[:] ratings
    cdef int n_users
    cdef int n_movies

    def __init__(self):
        data_reader = Movielens1MReader(0.8)
        URM_train = data_reader.URM_train
        URM_test = data_reader.URM_test
        self.users = data_reader.users
        self.movies = data_reader.movies
        self.ratings = data_reader.ratings
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]

        cdef int user
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
        cdef int[:] URM_without_indptr, t_column_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data
        cdef double [:, :] prediction, error
        cdef double[:, :] G
        cdef double gradient
        cdef double error_function
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


        #python passa le cose per riferimento, noi siamo interessati a copiarne i valori.
        #TODO: è lento perché stiamo cambiando i valori di una matrice sparsa

        # Needed for Adagrad
        G = np.zeros((self.n_movies, self.n_movies))
        eps = 1e-8

        URM_indices = URM_train.indices
        URM_data = URM_train.data
        URM_indptr = URM_train.indptr
        for j in range(1, self.n_movies):
            print("Column %s of %s" %(j, self.n_movies))
            URM_without = URM_train.copy()
            URM_without[:,j] = np.zeros((self.n_users,1))
            t_column_indices = csc_URM_train.indices[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]
            t_column_data = csc_URM_train.data[csc_URM_train_indptr[j]:csc_URM_train_indptr[j+1]]
            t_column = URM_train[:, j]
            for n_iter in range(iterations):
                if n_iter % 100 == 0:
                    print("Iteration #%s" %(n_iter))

                counter = 0
                start_time = time.time()
                for t_index in range(len(t_column_indices)):
                    user = t_column_indices[t_index]

                    URM_vector_indices = URM_indices[URM_indptr[user]:URM_indptr[user+1]]
                    URM_vector_data = URM_data[URM_indptr[user]:URM_indptr[user+1]]
                    partial_error = (cython_product_sparse(URM_vector_indices, URM_vector_data, S[:, j]) - t_column_data[counter])

                    for index in range(len(URM_vector_indices)):

                        gradient = partial_error*URM_vector_data[index] + gamma + beta*S[URM_vector_indices[index], j]
                        G[URM_vector_indices[index], j] += gradient**2
                        S[URM_vector_indices[index], j] -= (alpha/sqrt(G[URM_vector_indices[index], j] + eps))*gradient
                        if S[URM_vector_indices[index], j] < 0:
                            S[URM_vector_indices[index], j] = 0
                    counter = counter + 1

                error_function = np.linalg.norm(cython_product_t_column(URM_without, S[:, j], t_column_indices), 2)**2 + beta*np.linalg.norm(S[:, j], 2)**2  + gamma*np.linalg.norm(S[:, j], 1)

                if error_function < threshold:
                    break
                if n_iter % 100 == 0:
                    print("error function is: ", error_function)

            #print prediction for all the values different from zero.
            for i in range(self.n_users):
                if (URM_train[i, j] != 0):
                    print("Real: %s    predicted: %s" %(URM_train[i, j], cython_product_sparse(URM_indices[URM_indptr[i]:URM_indptr[i+1]],URM_data[URM_indptr[i]:URM_indptr[i+1]], S[:, j])))

