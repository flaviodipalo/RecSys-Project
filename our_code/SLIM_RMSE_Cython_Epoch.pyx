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

cdef double[:] clean_support_vector(double[:] vector):

    cdef int i

    for i in range(vector.shape[0]):
        vector[i] = 0

    return vector

#@cython.boundscheck(False)
cdef class SLIM_RMSE_Cython_Epoch:

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
    cdef int[:] S_indices, S_indptr
    cdef double[:] S_data


    #Adagrad
    cdef double [:, :] adagrad_cache


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
        self.adam_m = np.zeros((2, 2))                  #CAMBIARE
        self.adam_v = np.zeros((2, 2))                  #CAMBIARE
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.time_t = 0

        #RMSprop
        self.rms_prop_term = np.zeros((2, 2))           #CAMBIARE


        #GRADIENT DESCENT EPS FOR AVOIDING DIVISION BY ZERO
        self.eps = 10e-8

        S = sp.sparse.random(self.n_movies, self.n_movies, format='csc', density=0.001)
        self.S_indices = S.indices
        self.S_indptr = S.indptr
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
        self.rows = np.zeros(self.n_movies)
        #self.cols = np.zeros(self.n_movies)
        self.vals = np.zeros(self.n_movies)
        #cdef double[:] rows = self.rows
        #cdef double[:] cols = self.cols
        cdef double[:] vals = self.vals
        cdef long[:] topK_elements_indices
        cdef double value_to_insert
        cdef int index_to_use


        ##### PARTE MATRICE SPARSA
        cdef double[:, :] support_matrix_values = np.zeros((self.n_movies, self.topK))
        cdef long[:, :] support_matrix_indices = np.zeros((self.n_movies, self.topK)).astype(long)



        if gradient_option == "adam":
            self.time_t += 1


        total_normalization_error = 0
        #for j in self.unique_movies:




        for j in range(0, n_movies):
            gradient_vector = 0
            #rows = clean_support_vector(rows)
            #cols = clean_support_vector(cols)
            vals = clean_support_vector(vals)
            if j%1 == 0:
                print(j, n_movies)

            if self.similarity_matrix_normalized:
                sum_vector = vector_sum(S_data)
                for index in range(S_data.shape[0]):
                    S_data[index] /= sum_vector

            counter = 0
            for t_index in range(all_items_indices[all_items_indptr[j]:all_items_indptr[j+1]].shape[0]):

                user_index = all_items_indices[all_items_indptr[j]:all_items_indptr[j+1]][t_index]
                if URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0] > 1:
                    partial_error = (cython_product_sparse(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]], URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]], S_indices[S_indptr[j]:S_indptr[j+1]], S_data[S_indptr[j]:S_indptr[j+1]], j) - all_items_data[all_items_indptr[j]:all_items_indptr[j+1]][counter])
                    cum_loss += partial_error**2

                    if self.similarity_matrix_normalized:

                        length = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0] - 1

                        diagonal_value_P = 1 - (1/<double>length)
                        other_value_P = - (1/<double>length)

                        non_zero_gradient = <double *>PyMem_Malloc((URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0] - 1) * sizeof(double ))

                        support_index = 0
                        for index in range(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0]):
                            target_user_index = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index]
                            if target_user_index != j:
                                found = False
                                for index_for_found_flag in range(S_indices[S_indptr[j]:S_indptr[j+1]].shape[0]):
                                    if S_indices[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] == target_user_index:
                                        found = True
                                        break
                                non_zero_gradient[support_index] = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index] + i_beta*S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] + i_gamma
                                adagrad_cache[target_user_index, j] =  non_zero_gradient[support_index]**2


                                if gradient_option == "adagrad":
                                    adagrad_cache[target_user_index, j] += (non_zero_gradient[support_index])**2
                                    non_zero_gradient[support_index] = (1/sqrt(adagrad_cache[target_user_index, j] + eps))*non_zero_gradient[support_index]

                                elif gradient_option == "adam":
                                    self.adam_m[target_user_index, j] = self.beta_1*self.adam_m[target_user_index, j] + (1-self.beta_1)*non_zero_gradient[support_index]
                                    self.adam_v[target_user_index, j] = self.beta_2*self.adam_v[target_user_index, j] + (1-self.beta_2)*(non_zero_gradient[support_index])**2
                                    self.m_adjusted = self.adam_m[target_user_index, j]/(1 - self.beta_1**self.time_t)
                                    self.v_adjusted = self.adam_v[target_user_index, j]/(1 - self.beta_2**self.time_t)
                                    non_zero_gradient[support_index] = self.m_adjusted/(sqrt(self.v_adjusted) + eps)

                                elif gradient_option == "rmsprop":
                                    self.rms_prop_term[target_user_index, j] = 0.9*self.rms_prop_term[target_user_index,j] + 0.1*non_zero_gradient[support_index]**2
                                    non_zero_gradient[support_index] = non_zero_gradient[support_index]/(sqrt(self.rms_prop_term[target_user_index,j] + eps))

                                support_index += 1


                    sum_gradient = vector_sum(adagrad_cache[:, j])
                    p_index = 0
                    for index in range(URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]].shape[0]):
                        target_user_index = URM_indices[URM_indptr[user_index]:URM_indptr[user_index+1]][index]

                        if target_user_index != j:

                            if self.similarity_matrix_normalized:
                                gradient = vector_product(diagonal_value_P, other_value_P, non_zero_gradient, j, p_index, length)
                                gradient_vector += gradient
                                p_index += 1
                                if found:
                                    vals[target_user_index] += S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] - alpha*gradient
                                else:
                                    vals[target_user_index] -= alpha*gradient
                                if vals[target_user_index] < 0:
                                        vals[target_user_index] = 0
                            else:
                                found = False
                                for index_for_found_flag in range(S_indices[S_indptr[j]:S_indptr[j+1]].shape[0]):
                                    if S_indices[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] == target_user_index:
                                        found = True
                                        break
                                if found:
                                    gradient = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index] + i_beta*S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] + i_gamma
                                else:
                                    gradient = partial_error*URM_data[URM_indptr[user_index]:URM_indptr[user_index+1]][index] + i_gamma
                                if gradient_option == adagrad_option:
                                    adagrad_cache[target_user_index, j] += gradient**2
                                    #print(index_for_support, rows.shape[0])
                                    if found:
                                        vals[target_user_index] += S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] - (alpha/sqrt(adagrad_cache[target_user_index, j] + eps))*gradient
                                        #rows[target_user_index] = target_user_index
                                        #cols[index_for_support] = j
                                    else:
                                        vals[target_user_index] -= (alpha/sqrt(adagrad_cache[target_user_index, j] + eps))*gradient

                                    if vals[target_user_index] < 0:
                                        vals[target_user_index] = 0

                                elif gradient_option == normal_option:
                                    if found:
                                        vals[target_user_index] += S_data[S_indptr[j]:S_indptr[j+1]][index_for_found_flag] - alpha*gradient
                                    else:
                                        vals[target_user_index] -= alpha*gradient

                            '''
                            elif gradient_option == adam_option:
                                self.adam_m[target_user_index, j] = self.beta_1*self.adam_m[target_user_index, j] + (1-self.beta_1)*gradient
                                self.adam_v[target_user_index, j] = self.beta_2*self.adam_v[target_user_index, j] + (1-self.beta_2)*(gradient)**2
                                self.m_adjusted = self.adam_m[target_user_index, j]/(1 - self.beta_1**self.time_t)
                                self.v_adjusted = self.adam_v[target_user_index, j]/(1 - self.beta_2**self.time_t)
                                S[target_user_index, j] -= alpha*self.m_adjusted/(sqrt(self.v_adjusted) + eps)

                            elif gradient_option == rmsprop_option:
                                self.rms_prop_term[target_user_index,j] = 0.9*self.rms_prop_term[target_user_index,j] + 0.1*gradient**2
                                S[target_user_index, j] -= alpha*gradient/(sqrt(self.rms_prop_term[target_user_index,j] + eps))


                            '''


                    counter += 1
                    if self.similarity_matrix_normalized:
                        PyMem_Free(non_zero_gradient)

            '''
            if self.similarity_matrix_normalized:
                #print("SUM", j, vector_sum(S[:, j]))

                total_normalization_error += vector_sum(S[:, j]) - 1
                S[j, j] = 0
                sum_vector = vector_sum(S[:, j])
                for index in range(S[:, j].shape[0]):
                    S[index, j] /= sum_vector
            '''

            #### SORTING ####
            value_to_insert = 0
            for index_for_support in range(vals.shape[0]):
                if vals[index_for_support] != 0:
                    value_to_insert += 1
            #print("NON 0 ELEMENTS", value_to_insert)
            topK_elements_indices = np.argpartition(vals, -self.topK)[-self.topK:]
            support_matrix_indices[j, :] = topK_elements_indices
            for index_for_support in range(self.topK):
                #print(vals[support_matrix_indices[j, index_for_support]])
                index_to_use = int(support_matrix_indices[j, index_for_support])
                value_to_insert = vals[index_to_use]
                support_matrix_values[j, index_for_support] = value_to_insert

        print("Creating S matrix...")
        rows, cols, values = [], [], []
        for j in range(self.n_movies):
            for index_for_support in range(self.topK):
                value_to_insert = support_matrix_values[j, index_for_support]
                if value_to_insert > 0:
                    rows.append(support_matrix_indices[j, index_for_support])
                    cols.append(j)
                    values.append(value_to_insert)

        S_matrix = sp.sparse.csc_matrix((values, (rows, cols)), shape=(self.n_movies, self.n_movies))
        self.S_data = S_matrix.data
        self.S_indices = S_matrix.indices
        self.S_indptr = S_matrix.indptr
        print(cum_loss)
        #self.S = S
        self.adagrad_cache = adagrad_cache


    def get_S(self):

        S = sp.sparse.csc_matrix((self.S_data, self.S_indices, self.S_indptr), shape=(self.n_movies, self.n_movies))
        return S


##################################################################################################################
#####################
#####################            SPARSE MATRIX
#####################
##################################################################################################################

import scipy.sparse as sps

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free#, qsort

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil


# Node struct
ctypedef struct matrix_element_tree_s:
    long column
    double data
    matrix_element_tree_s *higher
    matrix_element_tree_s *lower

ctypedef struct head_pointer_tree_s:
    matrix_element_tree_s *head


# Function to allocate a new node
cdef matrix_element_tree_s * pointer_new_matrix_element_tree_s(long column, double data, matrix_element_tree_s *higher,  matrix_element_tree_s *lower):

    cdef matrix_element_tree_s * new_element

    new_element = < matrix_element_tree_s * > malloc(sizeof(matrix_element_tree_s))
    new_element.column = column
    new_element.data = data
    new_element.higher = higher
    new_element.lower = lower

    return new_element


# Functions to compare structs to be used in C qsort
cdef int compare_struct_on_column(const void *a_input, const void *b_input):
    """
    The function compares the column contained in the two struct passed.
    If a.column > b.column returns >0
    If a.column < b.column returns <0

    :return int: a.column - b.column
    """

    cdef head_pointer_tree_s *a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s *b_casted = <head_pointer_tree_s *> b_input

    return a_casted.head.column  - b_casted.head.column



cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0
    If a.data < b.data returns <0

    :return int: +1 or -1
    """

    cdef head_pointer_tree_s * a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s * b_casted = <head_pointer_tree_s *> b_input

    if (a_casted.head.data - b_casted.head.data) > 0.0:
        return +1
    else:
        return -1



#################################
#################################       CLASS DECLARATION
#################################

cdef class Sparse_Matrix_Tree_CSR:

    cdef long num_rows, num_cols

    # Array containing the struct (object, not pointer) corresponding to the root of the tree
    cdef head_pointer_tree_s* row_pointer

    def __init__(self, long num_rows, long num_cols):

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.row_pointer = < head_pointer_tree_s *> malloc(self.num_rows * sizeof(head_pointer_tree_s))

        # Initialize all rows to empty
        for index in range(self.num_rows):
            self.row_pointer[index].head = NULL


    cpdef double set_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.

        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        cdef matrix_element_tree_s* current_element, new_element, * old_element
        cdef int stopSearch = False


        # If the row is empty, create a new element
        if self.row_pointer[row].head == NULL:

            # row_pointer is a python object, so I need the object itself and not the address
            self.row_pointer[row].head = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True

        # If the cell exist, update its value
        if current_element.column == col:
            current_element.data = value

            return current_element.data


        # The cell is not found, create new Higher element
        elif current_element.column < col and current_element.higher == NULL:

            current_element.higher = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        # The cell is not found, create new Lower element
        elif current_element.column > col and current_element.lower == NULL:

            current_element.lower = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        else:
            assert False, 'ERROR - Current insert operation is not implemented'


    cpdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.

        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        cdef matrix_element_tree_s* current_element, new_element, * old_element
        cdef int stopSearch = False


        # If the row is empty, create a new element
        if self.row_pointer[row].head == NULL:

            # row_pointer is a python object, so I need the object itself and not the address
            self.row_pointer[row].head = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True

        # If the cell exist, update its value
        if current_element.column == col:
            current_element.data += value

            return current_element.data


        # The cell is not found, create new Higher element
        elif current_element.column < col and current_element.higher == NULL:

            current_element.higher = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        # The cell is not found, create new Lower element
        elif current_element.column > col and current_element.lower == NULL:

            current_element.lower = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        else:
            assert False, 'ERROR - Current insert operation is not implemented'




    cpdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.

        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """


        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError(
                "Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                    self.num_rows, self.num_cols, row, col))


        cdef matrix_element_tree_s* current_element
        cdef int stopSearch = False

        # If the row is empty, return default
        if self.row_pointer[row].head == NULL:
            return 0.0


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True


        # If the cell exist, return its value
        if current_element.column == col:
            return current_element.data

        # The cell is not found, return default
        else:
            return 0.0




    cpdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object

        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row

        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # row contains data
            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)


                # Flatten the tree data
                subtree_column, subtree_data = self.from_linked_list_to_python_list(self.row_pointer[row].head)
                data.extend(subtree_data)
                indices.extend(subtree_column)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))



    cpdef rebalance_tree(self, long TopK = False):
        """
        The function builds a balanced binary tree from the current one, for all matrix rows

        :param TopK: either False or an integer number. Number of the highest elements to preserve
        """

        cdef long row

        #start_time = time.time()

        for row in range(self.num_rows):

            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)
















    cdef matrix_element_tree_s * subtree_to_list_flat(self, matrix_element_tree_s * root):
        """
        The function flatten the structure of the subtree whose root is passed as a paramether
        The list is bidirectional and ordered with respect to the column
        The column ordering follows from the insertion policy

        :param root: tree root
        :return list, list: data and corresponding column. Empty list if root is None
        """

        if root == NULL:
            return NULL

        cdef matrix_element_tree_s *flat_list_head, *current_element

        # Flatten lower subtree
        flat_list_head = self.subtree_to_list_flat(root.lower)

        # If no lower elements exist, the head is the current element
        if flat_list_head == NULL:
            flat_list_head = root
            root.lower = NULL

        # Else move to the tail and add the subtree root
        else:
            current_element = flat_list_head
            while current_element.higher != NULL:
                current_element = current_element.higher

            # Attach the element with the bidirectional pointers
            current_element.higher = root
            root.lower = current_element

        # Flatten higher subtree and attach it to the tail of the flat list
        root.higher = self.subtree_to_list_flat(root.higher)

        # Attach the element with the bidirectional pointers
        if root.higher != NULL:
            root.higher.lower = root

        return flat_list_head



    cdef from_linked_list_to_python_list(self, matrix_element_tree_s * head):

        data = []
        column = []

        while head != NULL:
            data.append(head.data)
            column.append(head.column)

            head = head.higher

        return column, data



    cdef subtree_free_memory(self, matrix_element_tree_s* root):
        """
        The function frees all struct in the subtree whose root is passed as a parameter, root included

        :param root: tree root
        """

        if root != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(root.higher)
            self.subtree_free_memory(root.lower)

            # Once the lower elements have been reached, start freeing from the bottom
            free(root)



    cdef list_free_memory(self, matrix_element_tree_s * head):
        """
        The function frees all struct in the list whose head is passed as a parameter, head included

        :param head: list head
        """

        if head != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(head.higher)

            # Once the tail element have been reached, start freeing from them
            free(head)



    cdef matrix_element_tree_s* build_tree_from_list_flat(self, matrix_element_tree_s* flat_list_head):
        """
        The function builds a tree containing the passed data. This is the recursive function, the
        data should be sorted by te caller
        To ensure the tree is balanced, data is sorted according to the column

        :param row: row in which to create new tree
        :param column_vector: column coordinates
        :param data_vector: cell data
        """

        if flat_list_head == NULL:
            return NULL


        cdef long list_length = 0
        cdef long middle_element_step = 0

        cdef matrix_element_tree_s *current_element, *middleElement, *tree_root

        current_element = flat_list_head
        middleElement = flat_list_head

        # Explore the flat list moving the middle elment every tho jumps
        while current_element != NULL:
            current_element = current_element.higher
            list_length += 1
            middle_element_step += 1

            if middle_element_step == 2:
                middleElement = middleElement.higher
                middle_element_step = 0

        tree_root = middleElement

        # To execute the recursion it is necessary to cut the flat list
        # The last of the lower elements will have to be a tail
        if middleElement.lower != NULL:
            middleElement.lower.higher = NULL

            tree_root.lower = self.build_tree_from_list_flat(flat_list_head)


        # The first of the higher elements will have to be a head
        if middleElement.higher != NULL:
            middleElement.higher.lower = NULL

            tree_root.higher = self.build_tree_from_list_flat(middleElement.higher)


        return tree_root




    cdef matrix_element_tree_s* topK_selection_from_list(self, matrix_element_tree_s* head, long TopK):
        """
        The function selects the topK highest elements in the given list

        :param head: head of the list
        :param TopK: number of highest elements to preserve
        :return matrix_element_tree_s*: head of the new list
        """

        cdef head_pointer_tree_s *vector_pointer_to_list_elements
        cdef matrix_element_tree_s *current_element
        cdef long list_length, index, selected_count

        # Get list size
        current_element = head
        list_length = 0

        while current_element != NULL:
            list_length += 1
            current_element = current_element.higher


        # If list elements are not enough to perform a selection, return
        if list_length < TopK:
            return head

        # Allocate vector that will be used for sorting
        vector_pointer_to_list_elements = < head_pointer_tree_s *> malloc(list_length * sizeof(head_pointer_tree_s))

        # Fill vector wit pointers to list elements
        current_element = head
        for index in range(list_length):
            vector_pointer_to_list_elements[index].head = current_element
            current_element = current_element.higher


        # Sort array elements on their data field
        qsort(vector_pointer_to_list_elements, list_length, sizeof(head_pointer_tree_s), compare_struct_on_data)

        # Sort only the TopK according to their column field
        # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
        qsort(&vector_pointer_to_list_elements[list_length-TopK], TopK, sizeof(head_pointer_tree_s), compare_struct_on_column)


        # Rebuild list attaching the consecutive elements
        index = list_length-TopK

        # Detach last TopK element from previous ones
        vector_pointer_to_list_elements[index].head.lower = NULL

        while index<list_length-1:
            # Rearrange bidirectional pointers
            vector_pointer_to_list_elements[index+1].head.lower = vector_pointer_to_list_elements[index].head
            vector_pointer_to_list_elements[index].head.higher = vector_pointer_to_list_elements[index+1].head

            index += 1

        # Last element in vector will be the hew head
        vector_pointer_to_list_elements[list_length - 1].head.higher = NULL

        # Get hew list head
        current_element = vector_pointer_to_list_elements[list_length-TopK].head

        # If there are exactly enough elements to reach TopK, index == 0 will be the tail
        # Else, index will be the tail and the other elements will be removed
        index = list_length - TopK - 1
        if index > 0:

            index -= 1
            while index >= 0:
                free(vector_pointer_to_list_elements[index].head)
                index -= 1

        # Free array
        free(vector_pointer_to_list_elements)


        return current_element
