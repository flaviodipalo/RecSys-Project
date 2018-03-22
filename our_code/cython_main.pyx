from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
import numpy as np
import math
import random
import scipy.sparse as sps
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
            #TODO: questo è impostato per essere solo su una colonna
            cdef int j = 0

            for i in t_column_indices:
                URM_without_indptr = URM_without.indptr
                URM_without_indices = URM_without.indices[URM_without_indptr[i]:URM_without_indptr[i+1]]
                URM_without_data = URM_without.data[URM_without_indptr[i]:URM_without_indptr[i + 1]]
                for x in range(len(URM_without_data)):
                    prediction[i, j] += URM_without_data[x]*S[URM_without_indices[x], 0]
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
        #TODO: j è fisso per ora siccome abbiamo considerato una sola colonna
        cdef int j = 1
        cdef double alpha = 1e-3
        cdef int gamma = 1
        cdef double beta = 1e-2
        cdef int iterations = 1000
        cdef int threshold = 5
        cdef double[:,:] S = np.random.rand(self.n_movies,1)
        cdef int[:] URM_without_indptr, URM_without_indices, URM_without_data, t_column_indices
        cdef double[:] t_column_data
        cdef double [:, :] prediction, error
        cdef double gradient
        cdef double error_function
        cdef double [:, :] max_arg_s = np.zeros((iterations, 1))

        #Adam
        adam = True
        cdef double[:, :] m = np.zeros((iterations, 1))
        cdef double[:, :] v = np.zeros((iterations, 1))
        cdef double beta1 = 0.9
        cdef double beta2 = 0.999
        cdef double eps_adam = 1e-8

        #Adagrad
        adagrad = False
        cdef double[:, :] G = np.zeros((self.n_movies, 1))
        cdef double eps_adagrad = 1e-8



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


        start_time = time.time()
        for n_iter in range(1, iterations):
            print("Iteration #%s" %(n_iter))
            if n_iter%100 == 0:
                for i, e in enumerate(t_column_data):
                    if e != 0:
                        prediction[i, 0] = self.cython_product_sparse(URM_without[i, :], S)

                print("the sum of the element is: ", self.vector_sum(prediction[:, 0]))

                for i in range(t_column_data.shape[0]):
                    error[i, 0] = prediction[i, 0] - t_column_data[i]
                print("the sum of the errors is: ", self.vector_sum(error[:, 0]))

            i = n_iter%self.n_users
            for gradient_index in range (self.n_movies):
                gradient = (self.cython_product_sparse(URM_without[i, :], S) - t_column[i, 0])*URM_without[i, gradient_index] + gamma + beta*S[gradient_index, 0]
                if adagrad:
                    G[gradient_index, 0] += gradient**2
                    S[gradient_index, 0] -= (alpha/math.sqrt(G[gradient_index, 0] + eps_adagrad))*gradient
                if adam:
                    m[n_iter, 0] = beta1 * m[n_iter-1, 0] + (1 - beta1)*gradient
                    v[n_iter, 0] = beta2 * v[n_iter-1, 0] + (1 - beta2)*(gradient)**2
                    m_hat = v[n_iter, 0]/(1-beta1**n_iter)
                    v_hat = v[n_iter, 0]/(1-beta2**n_iter)
                    S[gradient_index, 0] -= alpha * (m_hat/(math.sqrt(v_hat) + eps_adam))
                if S[gradient_index, 0] < 0:
                    S[gradient_index, 0] = 0

            '''
            for i in t_column_indices:
                #TODO: problem here in computing the gradient.
                ##no sgd
                for gradient_index in range (self.n_movies):
                    gradient = (self.cython_product_sparse(URM_without[i, :], S) - t_column[i, 0])*URM_without[i, gradient_index] + gamma + beta*S[gradient_index, 0]
                    G[gradient_index, 0] += gradient**2
                    S[gradient_index, 0] -= (alpha/math.sqrt(G[gradient_index, 0] + eps))*gradient
                    if S[gradient_index, 0] < 0:
                        S[gradient_index, 0] = 0

                gradient_index = random.randrange(0, self.n_movies-1, 1)
                #print('i, gradient:',i,gradient)
                gradient = (self.cython_product_sparse(URM_without[i, :], S) - t_column[i, 0])*URM_without[i, gradient_index] + gamma + beta*S[gradient_index, 0]
                G[gradient_index, 0] += gradient**2
                S[gradient_index, 0] -= (alpha/math.sqrt(G[gradient_index, 0] + eps))*gradient
                if S[gradient_index, 0] < 0:
                    S[gradient_index, 0] = 0
            '''
            error_function = self.cython_norm(self.cython_product_t_column(URM_without, S, t_column_indices), 2)**2 + beta*self.cython_norm(S, 2)**2  + gamma*self.cython_norm(S, 1)

            if error_function < threshold:
                break
            print('error function is: ',error_function)

        print("The total time for %s iterations is %s seconds" %(n_iter+1, time.time()-start_time))

        #print prediction for all the values different from zero.
        for i in range(self.n_users):
            if (URM_train[i, j] != 0):
                print("Real: %s    predicted: %s" %(URM_train[i, 1], self.cython_product_sparse(URM_without[i, :], S)))