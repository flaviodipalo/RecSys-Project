from libc.stdlib cimport malloc, free
import numpy as np
from cython.parallel import parallel, prange
import scipy as sp
import cython
from libc.stdio cimport printf
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.math cimport sqrt
import random
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))
#python setup.py build_ext --inplace

#TODO: usare il 10M MA usato così: una volta completo ed una volta togliendo il 33% dei top popular item.
#TODO: utilizziamo un KNN come baseline
#TODO: proviamo

#@cython.boundscheck(False)
cdef double vector_product(double diagonal_value, double other_value,double * B, int column, int p_index, int length):


    cdef int i
    cdef long double result

    result = 0
    for i in range(length):
        if i == p_index:
            result += diagonal_value*B[i]
        else:
            result += other_value*B[i]

    return result

#@cython.boundscheck(False)
cdef double vector_sum(double[:] vector):

    cdef int i
    cdef double counter = 0

    for i in range(vector.shape[0]):
        counter += vector[i]

    return counter

#@cython.boundscheck(False)
cdef double cython_product_sparse(int[:] URM_indices, double[:] URM_data, int[:] S_indices, double[:] S_data, int column_index_with_zero):

    cdef double result = 0
    cdef int x, j
    cdef bint found = False

    for x in range(URM_data.shape[0]):
        if URM_indices[x] != column_index_with_zero:
            for j in range(S_indices.shape[0]):
                if S_indices[j] == URM_indices[x]:
                    found = True
            if found:
                result += URM_data[x]*S_data[j]

    return result

cdef double[:] clean_support_vector(double[:] vector, int length):

    cdef int i

    for i in range(length):
        vector[i] = 0

    return vector

#@cython.boundscheck(False)
cdef class SLIM_RMSE_Cython_Epoch:

    cdef int[:] S_indptr
    cdef int[:] S_indices
    cdef double[:] S_data
    cdef double[:] users, movies, ratings
    cdef int[:] all_items_indptr, all_items_indices
    cdef double[:] all_items_data
    cdef int[:] URM_indptr,  URM_indices
    cdef double[:] URM_data
    #cdef double[:, :] S
    cdef int n_users, n_movies, i_gamma
    cdef double alpha
    cdef int i_iterations
    cdef double i_beta
    cdef str gradient_option
    cdef bint similarity_matrix_normalized
    cdef double **P
    cdef int topK
    cdef double[:] rows, cols, vals

    #Adagrad
    cdef double[:, :] adagrad_cache


    #Adam
    cdef double beta_1, beta_2, m_adjusted, v_adjusted
    cdef double[:, :] adam_m, adam_v
    cdef int time_t

    #RMSprop
    cdef double[:, :] rms_prop_term
    cdef double eps


    def __init__(self, URM_train, learning_rate, gamma, beta, iterations, gradient_option, similarity_matrix_normalized, topK=None):

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
        self.topK = topK

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
            S = np.random.rand( self.n_movies, self.n_movies)
        #else:
        #    self.S = np.zeros((self.n_movies, self.n_movies))

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


        S = sp.sparse.random(self.n_movies, self.n_movies, format='csc')
        self.S_indptr = S.indptr
        self.S_indices = S.indices
        self.S_data = S.data


    @cython.boundscheck(False)
    def epochIteration_Cython(self):

        cdef double[:] prediction = np.zeros(self.n_users)

        cdef int[:] URM_without_indptr, t_column_indices, item_indices
        cdef int[:, :] URM_without_indices, URM_without_data
        cdef double[:] t_column_data

        cdef double gradient, gradient_vector

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



        cdef int n_movies = self.n_movies
        #cdef double[:, :] S = np.zeros(5, 2)       #TOGLEIREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        #S = sp.sparse.random(self.n_movies, self.n_movies, format='csc')
        cdef int[:] all_items_indices = self.all_items_indices
        cdef int[:] all_items_indptr = self.all_items_indptr
        cdef int[:] URM_indices = self.URM_indices
        cdef int[:] URM_indptr = self.URM_indptr
        cdef double[:] URM_data = self.URM_data
        cdef double i_beta = self.i_beta
        cdef double i_gamma = self.i_gamma
        cdef double[:, :] adagrad_cache = self.adagrad_cache
        cdef double[:] all_items_data = self.all_items_data
        cdef double alpha = self.alpha
        cdef double eps = self.eps
        cdef str gradient_option = self.gradient_option
        cdef str adagrad_option = "adagrad"
        cdef str adam_option = "adam"
        cdef str rmsprop_option = "rmsprop"
        cdef str normal_option = "normal"
        cdef double sum_vector
        cdef int index_for_support
        cdef double diagonal_value_P, other_value_P
        cdef bint found
        cdef int index_for_found_flag
        cdef int[:] S_indptr = self.S_indptr
        cdef int[:] S_indices = self.S_indices
        cdef double[:] S_data = self.S_data
        self.rows = np.zeros(self.n_users)
        self.cols = np.zeros(self.n_users)
        self.vals = np.zeros(self.n_users)
        cdef double[:] rows = self.rows
        cdef double[:] cols = self.cols
        cdef double[:] vals = self.vals


        ##### PARTE MATRICE SPARSA
        cdef double[:, :] support_matrix_value = np.zeros((self.n_movies, self.topK))
        cdef double[:, :] support_matrix_indices = np.zeros((self.n_movies, self.topK))



        if gradient_option == "adam":
            self.time_t += 1


        total_normalization_error = 0
        #for j in self.unique_movies:




        for j in range(0, n_movies):
            gradient_vector = 0
            index_for_support = 0
            rows = clean_support_vector(rows, self.n_users)
            cols = clean_support_vector(cols, self.n_users)
            vals = clean_support_vector(vals, self.n_users)
            if j%1 == 0:
                print(j, n_movies)
            #S[j, j] = 0
            #########METTERE A POSTO PER SIMILARITA'################
            '''
            if self.similarity_matrix_normalized:
                sum_vector = vector_sum(S[:, j])
                for index in range(S[:, j].shape[0]):
                    S[index, j] /= sum_vector
            '''
            ###########################################
            print("INIZIO")
            #if j%500 ==0:
            #    print("Column ", j)
            #t_column_indices = item_indices[item_indptr[j]:item_indptr[j+1]]
            #t_column_data = item_data[item_indptr[j]:item_indptr[j+1]]

            #item_indices = all_items_indices[all_items_indptr[j]:all_items_indptr[j+1]]

            counter = 0
            for t_index in range(all_items_indices[all_items_indptr[j]:all_items_indptr[j+1]].shape[0]):

                user_index = all_items_indices[all_items_indptr[j]:all_items_indptr[j+1]][t_index]
                #user_indices = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]]
                if URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0] > 1:
                    #user_data = URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]]

                    partial_error = (cython_product_sparse(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]], URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]], S_indices[S_indptr[j]:S_indptr[j+1]], S_data[S_indptr[j]:S_indptr[j+1]], j) - all_items_data[all_items_indptr[j]:all_items_indptr[j+1]][counter])
                    cum_loss += partial_error**2
                    '''
                    if self.similarity_matrix_normalized:

                        length = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0] - 1

                        #P = <double **>malloc(length * sizeof(double*))
                        #support_index = 0
                        #for index_for_gil in range(length):
                        #    P[support_index] = <double *>malloc(length * sizeof(double))
                        #    support_index = support_index + 1
#
 #                       for i in range(length):
  #                           nogilsupport_index = 0
   #                         for index_for_gil in range(length):
    #                            if i == support_index:
     #                               P[i][support_index] = 1 - (1/<double>length)
      #                          else:
       ##                             P[i][support_index] = - (1/<double>length)
         #                       support_index = support_index + 1

                        diagonal_value_P = 1 - (1/<double>length)
                        other_value_P = - (1/<double>length)

                        non_zero_gradient = <double *>PyMem_Malloc((URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0] - 1) * sizeof(double ))

                        support_index = 0
                        for index in range(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0]):
                            target_user_index = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index]
                            if target_user_index != j:
                                non_zero_gradient[support_index] = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index] + i_beta*S[target_user_index, j] + i_gamma
                                adagrad_cache[target_user_index, j] += non_zero_gradient[support_index]**2


                           #     if gradient_option == "adagrad":
                            #        adagrad_cache[target_user_index, j] += (non_zero_gradient[support_index])**2
                             #       non_zero_gradient[support_index] = (1/sqrt(adagrad_cache[target_user_index, j] + eps))*non_zero_gradient[support_index]
#
 #                               elif gradient_option == "adam":
  #                                  self.adam_m[target_user_index, j] = self.beta_1*self.adam_m[target_user_index, j] + (1-self.beta_1)*non_zero_gradient[support_index]
   #                                 self.adam_v[target_user_index, j] = self.beta_2*self.adam_v[target_user_index, j] + (1-self.beta_2)*(non_zero_gradient[support_index])**2
    #####                            self.m_adjusted = self.adam_m[target_user_index, j]/(1 - self.beta_1**self.time_t)
         #                           self.v_adjusted = self.adam_v[target_user_index, j]/(1 - self.beta_2**self.time_t)
          #                          non_zero_gradient[support_index] = self.m_adjusted/(sqrt(self.v_adjusted) + eps)
#
 #                               elif gradient_option == "rmsprop":
  #                                  self.rms_prop_term[target_user_index, j] = 0.9*self.rms_prop_term[target_user_index,j] + 0.1*non_zero_gradient[support_index]**2
   #                                 non_zero_gradient[support_index] = non_zero_gradient[support_index]/(sqrt(self.rms_prop_term[target_user_index,j] + eps))
    #

                                #non_zero_gradient[support_index] = randint(10**5, 10**10)

                                #print("ERROR", partial_error, user_data[index], non_zero_gradient[support_index])
                                support_index = support_index + 1
                    '''
                    print("META'")
                    sum_gradient = vector_sum(adagrad_cache[:, j])
                    p_index = 0
                    for index in range(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0]):
                        target_user_index = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index]

                        if target_user_index != j:
                            if self.similarity_matrix_normalized:
                                gradient = vector_product(diagonal_value_P, other_value_P, non_zero_gradient, j, p_index, length)
                                gradient_vector += gradient
                                p_index = p_index + 1
                                #S[target_user_index, j] -= alpha*gradient
                            else:
                                found = False
                                for index_for_found_flag in range(S_indices[S_indptr[j]:S_indptr[j+1]].shape[0]):
                                    if S_indices[S_indptr[j]:S_indptr[j+1]].shape[0] == target_user_index:
                                        found = True
                                if found:
                                    gradient = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index] + i_beta*S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] + i_gamma
                                else:
                                    gradient = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index] + i_gamma

                            if gradient_option == adagrad_option:
                                if self.similarity_matrix_normalized:
                                    print("NO")
                                    #S[target_user_index, j] -= (alpha/sqrt(sum_gradient)/n_movies + eps)*gradient

                                else:
                                    adagrad_cache[target_user_index, j] += gradient**2
                                    rows[index_for_support] = target_user_index
                                    cols[index_for_support] = j
                                    if found:
                                        vals[index_for_support] = S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] - (alpha/sqrt(adagrad_cache[target_user_index, j] + eps))*gradient
                                    else:
                                        vals[index_for_support] = - (alpha/sqrt(adagrad_cache[target_user_index, j] + eps))*gradient
                                    if vals[index_for_support] < 0:
                                        vals[index_for_support] = 0
                                    index_for_support += 1
                                    #S[target_user_index, j] -= (alpha/sqrt(adagrad_cache[target_user_index, j] + eps))*gradient

                            '''
                            elif gradient_option == adam_option:
                                self.adam_m[target_user_index, j] = self.beta_1*self.adam_m[target_user_index, j] + (1-self.beta_1)*gradient
                                self.adam_v[target_user_index, j] = self.beta_2*self.adam_v[target_user_index, j] + (1-self.beta_2)*(gradient)**2
                                self.m_adjusted = self.adam_m[target_user_index, j]/(1 - self.beta_1**self.time_t)
                                self.v_adjusted = self.adam_v[target_user_index, j]/(1 - self.beta_2**self.time_t)
                                S[target_user_index, j] -= alpha*self.m_adjusted/(sqrt(self.v_adjusted) + eps)
#
                            elif gradient_option == rmsprop_option:
                                self.rms_prop_term[target_user_index,j] = 0.9*self.rms_prop_term[target_user_index,j] + 0.1*gradient**2
                                S[target_user_index, j] -= alpha*gradient/(sqrt(self.rms_prop_term[target_user_index,j] + eps))

                            elif gradient_option == normal_option:
                                S[target_user_index, j] -= alpha*gradient
                            '''
                        #if S[target_user_index, j] < 0:
                        #    S[target_user_index, j] = 0
                    print("FINE")
                    counter += 1
                    #print(gradient_vector)
                    #if self.similarity_matrix_normalized:
                     #   PyMem_Free(non_zero_gradient)
                      #  #for i in range(length):
                            #free(P[i])
                        #free(P)

            '''
            if self.similarity_matrix_normalized:
                #print("SUM", j, vector_sum(S[:, j]))

                total_normalization_error += vector_sum(S[:, j]) - 1
                S[j, j] = 0
                sum_vector = vector_sum(S[:, j])
                for index in range(S[:, j].shape[0]):
                    S[index, j] /= sum_vector
            '''
        print(cum_loss)
        #self.S = S
        self.adagrad_cache = adagrad_cache


    def get_S(self):
        S = sp.sparse.csc_matrix((self.S_data, self.S_indices, self.S_indptr), shape=(self.n_movies, self.n_movies))
        return np.asarray(S)
