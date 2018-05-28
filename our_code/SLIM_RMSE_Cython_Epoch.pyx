from data.movielens_1m.Movielens1MReader import Movielens1MReader
from cython.parallel import parallel,  prange
cimport cython
from libc.stdlib cimport malloc, free
import numpy as np
from random import randint

from libc.math cimport sqrt
import random
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
#python setup.py build_ext --inplace
import time
import timeit

#TODO: usare il 10M MA usato così: una volta completo ed una volta togliendo il 33% dei top popular item.
#TODO: utilizziamo un KNN come baseline
#TODO: proviamo

cdef double vector_product(double* A,double * B, int column, int index, int length):


    cdef int i
    cdef long double result

    result = 0
    for i in range(length):
        result += A[i]*B[i]

    return result

cdef double vector_sum(double[:] vector):

    cdef int i
    cdef double counter = 0

    for i in range(vector.shape[0]):
        counter += vector[i]

    return counter


cdef double[:, :] scalar_with_matrix(double scalar, double[:, :] A):

    cdef int i, j
    cdef double [:, :] result

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = A[i, j]*scalar

    return A


cdef matrix_product(double[:, :] A, double[:, :] B, double[:, :] C):

    cdef int i, j, x
    cdef double result

    for i in range(A.shape[0]):
        for x in range(B.shape[1]):
            result = 0
            for j in range(A.shape[1]):
                result += A[i, j]*B[j, x]
            C[i, x] = result


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
    cdef int[:] all_items_indptr, all_items_indices
    cdef double[:] all_items_data
    cdef int[:] URM_indptr,  URM_indices
    cdef double[:] URM_data
    cdef double[:, :] S
    cdef int n_users, n_movies, i_gamma
    cdef double alpha
    cdef int i_iterations
    cdef double i_beta
    cdef str gradient_option
    cdef bint similarity_matrix_normalized
    cdef double **P

    #Adagrad
    cdef double[:, :] adagrad_cache


    #Adam
    cdef double beta_1, beta_2, m_adjusted, v_adjusted
    cdef double[:, :] adam_m, adam_v
    cdef int time_t

    #RMSprop
    cdef double[:, :] rms_prop_term
    cdef double eps


    def __init__(self, URM_train, learning_rate, gamma, beta, iterations, gradient_option,similarity_matrix_normalized):

        self.i_beta = beta
        self.i_iterations = iterations
        self.n_users = URM_train.shape[0]
        self.n_movies = URM_train.shape[1]
        self.URM_indices = URM_train.indices
        self.URM_data = URM_train.data
        self.URM_indptr = URM_train.indptr
        self.i_gamma = gamma
        self.alpha = learning_rate
        self.gradient_option = gradient_option
        self.similarity_matrix_normalized = similarity_matrix_normalized

        #for i in range(URM_train.shape[0]):
         #   for j in range(URM_train.shape[1]):
          #      if URM_train[i, j] != 0:
           #         print(URM_train[i, j], i, j)

        csc_URM_train = URM_train.tocsc()
        self.all_items_indptr = csc_URM_train.indptr
        self.all_items_indices = csc_URM_train.indices
        self.all_items_data = csc_URM_train.data

        if self.similarity_matrix_normalized:
            np.random.seed(0)
            #self.S = np.random.normal(0, 5, (self.n_movies, self.n_movies))
            self.S = np.random.rand( self.n_movies, self.n_movies)
        else:
            self.S = np.zeros((self.n_movies, self.n_movies))

        #ADAGRAD
        self.adagrad_cache = np.zeros((self.n_movies, self.n_movies))

        #ADAM
        self.adam_m = np.zeros((self.n_movies, self.n_movies))
        self.adam_v = np.zeros((self.n_movies, self.n_movies))
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.time_t = 0

        #RMSprop
        self.rms_prop_term = np.zeros((self.n_movies, self.n_movies))


        #GRADIENT DESCENT EPS FOR AVOIDING DIVISION BY ZERO
        self.eps = 10e-8


    def epochIteration_Cython(self):

        cdef double[:] prediction = np.zeros(self.n_users)

        cdef int[:] URM_without_indptr, t_column_indices, item_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data

        cdef double gradient,gradient_vector

        cdef double error_function, total_normalization_error, sum_gradient
        cdef double partial_error, cum_loss = 0

        cdef int counter
        cdef int time_counter = 0
        cdef int[:] user_indices
        cdef double[:] user_data

        cdef int user_index, target_user_index, new_adagrad_perfect
        cdef int j
        cdef int i
        cdef int index
        cdef int n_iter
        cdef int t_index
        cdef int p_index, length, non_zero_count, support_index
        cdef double *non_zero_gradient



        if self.gradient_option == "adam":
            self.time_t += 1


        total_normalization_error = 0
        #for j in self.unique_movies:
        for j in range(0, self.n_movies):
            gradient_vector = 0
            if j % 100 == 0:
                print(j, self.n_movies)
            self.S[j, j] = 0
            if self.similarity_matrix_normalized:
                sum_vector = vector_sum(self.S[:, j])
                for index in range(self.S[:, j].shape[0]):
                    self.S[index, j] /= sum_vector
            #if j%500 ==0:
            #    print("Column ", j)
            #t_column_indices = item_indices[item_indptr[j]:item_indptr[j+1]]
            #t_column_data = item_data[item_indptr[j]:item_indptr[j+1]]

            item_indices = self.all_items_indices[self.all_items_indptr[j]:self.all_items_indptr[j+1]]

            counter = 0
            for t_index in range(item_indices.shape[0]):

                user_index = item_indices[t_index]
                user_indices = self.URM_indices[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]]
                if user_indices.shape[0] > 1:
                    user_data = self.URM_data[self.URM_indptr[user_index]:self.URM_indptr[user_index+1]]
                    partial_error = (cython_product_sparse(user_indices, user_data, self.S[:, j], j) - self.all_items_data[self.all_items_indptr[j]:self.all_items_indptr[j+1]][counter])
                    cum_loss += partial_error**2

                    if self.similarity_matrix_normalized:

                        length = user_indices.shape[0] - 1
                        self.P = <double **>malloc(length * sizeof(double*))
                        for support_index in range(length):
                            self.P[support_index] = <double *>malloc(length * sizeof(double))

                        for i in range(length):
                            for support_index in range(length):
                                if i == support_index:
                                    self.P[i][support_index] = 1 - (1/<double>length)
                                else:
                                    self.P[i][support_index] = - (1/<double>length)


                        non_zero_gradient = <double *>malloc((user_indices.shape[0] - 1) * sizeof(double ))

                        support_index = 0
                        prova_vector = []

                        for index in range(user_indices.shape[0]):
                            target_user_index = user_indices[index]
                            if target_user_index != j:
                                non_zero_gradient[support_index] = partial_error*user_data[index] + self.i_beta*self.S[target_user_index, j] + self.i_gamma
                                self.adagrad_cache[target_user_index, j] += non_zero_gradient[support_index]**2
                                '''
                                if self.gradient_option == "adagrad":
                                    self.adagrad_cache[target_user_index, j] += (non_zero_gradient[support_index])**2
                                    non_zero_gradient[support_index] = (1/sqrt(self.adagrad_cache[target_user_index, j] + self.eps))*non_zero_gradient[support_index]

                                elif self.gradient_option == "adam":
                                    self.adam_m[target_user_index, j] = self.beta_1*self.adam_m[target_user_index, j] + (1-self.beta_1)*non_zero_gradient[support_index]
                                    self.adam_v[target_user_index, j] = self.beta_2*self.adam_v[target_user_index, j] + (1-self.beta_2)*(non_zero_gradient[support_index])**2
                                    self.m_adjusted = self.adam_m[target_user_index, j]/(1 - self.beta_1**self.time_t)
                                    self.v_adjusted = self.adam_v[target_user_index, j]/(1 - self.beta_2**self.time_t)
                                    non_zero_gradient[support_index] = self.m_adjusted/(sqrt(self.v_adjusted) + self.eps)

                                elif self.gradient_option == "rmsprop":
                                    self.rms_prop_term[target_user_index, j] = 0.9*self.rms_prop_term[target_user_index,j] + 0.1*non_zero_gradient[support_index]**2
                                    non_zero_gradient[support_index] = non_zero_gradient[support_index]/(sqrt(self.rms_prop_term[target_user_index,j] + self.eps))
                                '''
                                #non_zero_gradient[support_index] = randint(10**5, 10**10)

                                #print("ERROR", partial_error, user_data[index], non_zero_gradient[support_index])
                                support_index += 1
                    sum_gradient = vector_sum(self.adagrad_cache[:, j])
                    p_index = 0
                    for index in range(user_indices.shape[0]):
                        target_user_index = user_indices[index]

                        if target_user_index == j:
                            p_index -= 1

                        else:
                            if self.similarity_matrix_normalized:
                                gradient = vector_product(self.P[p_index], non_zero_gradient, j, user_index, length)
                                gradient_vector += gradient
                                #self.S[target_user_index, j] -= self.alpha*gradient
                            else:
                                gradient = partial_error*user_data[index] + self.i_beta*self.S[target_user_index, j] + self.i_gamma

                            if self.gradient_option == "adagrad":

                                self.S[target_user_index, j] -= (self.alpha/sqrt(sum_gradient)/len(self.adagrad_cache) + self.eps)*gradient

                                #self.S[target_user_index, j] -= (self.alpha/sqrt(self.adagrad_cache[target_user_index, j] + self.eps))*gradient


                            elif self.gradient_option == "adam":
                                self.adam_m[target_user_index, j] = self.beta_1*self.adam_m[target_user_index, j] + (1-self.beta_1)*gradient
                                self.adam_v[target_user_index, j] = self.beta_2*self.adam_v[target_user_index, j] + (1-self.beta_2)*(gradient)**2
                                self.m_adjusted = self.adam_m[target_user_index, j]/(1 - self.beta_1**self.time_t)
                                self.v_adjusted = self.adam_v[target_user_index, j]/(1 - self.beta_2**self.time_t)
                                self.S[target_user_index, j] -= self.alpha*self.m_adjusted/(sqrt(self.v_adjusted) + self.eps)
#
                            elif self.gradient_option == "rmsprop":
                                self.rms_prop_term[target_user_index,j] = 0.9*self.rms_prop_term[target_user_index,j] + 0.1*gradient**2
                                self.S[target_user_index, j] -= self.alpha*gradient/(sqrt(self.rms_prop_term[target_user_index,j] + self.eps))

                            elif self.gradient_option == "normal":
                                self.S[target_user_index, j] -= self.alpha*gradient

                        if self.S[target_user_index, j] < 0:
                            self.S[target_user_index, j] = 0
                        p_index += 1
                    counter += 1
                    #print(gradient_vector)
                    if self.similarity_matrix_normalized:
                        free(non_zero_gradient)
                        for i in range(length):
                            free(self.P[i])
                        free(self.P)


            if self.similarity_matrix_normalized:
                #print("SUM", j, vector_sum(self.S[:, j]))

                total_normalization_error += vector_sum(self.S[:, j]) - 1
                self.S[j, j] = 0
                sum_vector = vector_sum(self.S[:, j])
                for index in range(self.S[:, j].shape[0]):
                    self.S[index, j] /= sum_vector
        print("CUM loss: {:.2E}".format(cum_loss))

        #error_function = cython_norm(prediction_error(self.URM_indptr, self.URM_indices, self.URM_data, self.S[:, j], self.all_items_indices[self.all_items_indptr[j]:self.all_items_indptr[j+1]], self.all_items_data[self.all_items_indptr[j]:self.all_items_indptr[j+1]], j, prediction), 2)**2 + self.i_beta*cython_norm(self.S[:, j], 2)**2  + self.i_gamma*cython_norm(self.S[:, j], 1)
        #print(error_function)
        print("TOTAL", total_normalization_error)
#
    def get_S(self):
        return np.asarray(self.S)