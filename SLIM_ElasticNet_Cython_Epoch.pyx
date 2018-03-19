"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from Base.Recommender_utils import similarityMatrixTopK, check_matrix
import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct SGD_sample:
    long user
    long item
    double value

cdef class SLIM_ElasticNet_Cython:

    cdef int n_users
    cdef int n_items
    cdef int numPositiveIteractions
    cdef int topK
    cdef int useAdaGrad, rmsprop

    cdef float learning_rate

    cdef long[:] eligibleUsers
    cdef long numEligibleUsers

    cdef int[:] seenItemsSampledUser
    cdef int numSeenItemsSampledUser

    cdef int[:] URM_train_indices, URM_train_indptr
    cdef double[:], URM_train_data


    cdef S_sparse
    cdef double[:,:] S_dense


    def __init__(self, URM_train,
                 learning_rate = 0.05,
                 topK=100, sgd_mode='adagrad'):

        super(SLIM_ElasticNet_Cython, self).__init__()

        URM_train = check_matrix(URM_train, 'csr')

        self.numPositiveIteractions = int(URM_train.nnz * 1)
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.topK = topK

        self.URM_train_indices = URM_train.indices
        self.URM_train_indptr = URM_train.indptr
        self.URM_train_data = np.array(URM_train.data, dtype=float)


        if sgd_mode=='adagrad':
            self.useAdaGrad = True
        elif sgd_mode=='rmsprop':
            self.rmsprop = True
        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop'. Provided value was '{}'".format(
                    sgd_mode))

        self.learning_rate = learning_rate

    # Using memoryview instead of the sparse matrix itself allows for much faster access
    ## questo tipo di indirizzamento viene usato per evitare le matrici sparse in cython
    cdef int[:] getSeenItems(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]

    def epochIteration_Cython(self):

        cdef long start_time_epoch = time.time()
        cdef long start_time_batch = time.time()

        cdef SGD_sample sample
        cdef long index, seenItem, numCurrentSample, itemId
        cdef float gradient

        cdef int numSeenItems
        cdef int printStep

        if self.sparse_weights:
            printStep = 500000
        else:
            printStep = 5000000

        # Variables for AdaGrad and RMSprop
        cdef double [:] sgd_cache
        cdef double cacheUpdate
        cdef float gamma

        if self.useAdaGrad:
            sgd_cache = np.zeros((self.n_items), dtype=float)

        elif self.rmsprop:
            sgd_cache = np.zeros((self.n_items), dtype=float)
            gamma = 0.001

        # Uniform user sampling without replacement
        for numCurrentSample in range(self.numPositiveIteractions):

            #seleziona un random utente e per quell'utente un random film.
            sample = self.sampleBatch_Cython()

            x_uij = 0.0

            # The difference is computed on the user_seen items

            index = 0
            ##numero di item visti dall'utente.
            while index<self.numSeenItemsSampledUser:
                #array di indici degli elementi visti dall'utente.
                seenItem = self.seenItemsSampledUser[index]
                index +=1

                #print("Get: i {}, j {}, seenItem {}".format(i, j, seenItem))

                if self.sparse_weights:

                   x_uij += self.S_sparse.get_value(i, seenItem) - self.S_sparse.get_value(j, seenItem)
                else:
                    x_uij += self.S_dense[i, seenItem] - self.S_dense[j, seenItem]

            #TODO: non è questo il gradient che vogliamo
            gradient = 1 / (1 + exp(x_uij))

            if self.useAdaGrad:
                cacheUpdate = gradient ** 2

                sgd_cache[i] += cacheUpdate
                sgd_cache[j] += cacheUpdate
                #TODO: da sostituire il nuovo gradient
                gradient = gradient / (sqrt(sgd_cache[i]) + 1e-8)

            index = 0
            while index < self.numSeenItemsSampledUser:
                seenItem = self.seenItemsSampledUser[index]
                index +=1
                #TODO: come è fatto questo update del gradient ???
                if self.sparse_weights:

                    if seenItem != i:
                        self.S_sparse.add_value(i, seenItem, self.learning_rate * gradient)

                    if seenItem != j:
                        self.S_sparse.add_value(j, seenItem, -self.learning_rate * gradient)

                else:

                    if seenItem != i:
                        self.S_dense[i, seenItem] += self.learning_rate * gradient

                    if seenItem != j:
                        self.S_dense[j, seenItem] -= self.learning_rate * gradient

            #TODO: queste funzioni sono utili ma dopo
            '''
            # If I have reached at least 20% of the total number of batches or samples
            if numCurrentBatch % (totalNumberOfBatch/5) == 0 and numCurrentBatch!=0:
                self.S_sparse.rebalance_tree(TopK=self.topK)
                #print("Num batch is {}, rebalancing matrix".format(numCurrentBatch))


            if((numCurrentBatch%printStep==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/self.numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()
            '''

        # FIll diagonal with zeros

        index = 0
        while index < self.n_items:

            if self.sparse_weights:
                self.S_sparse.add_value(index, index, -self.S_sparse.get_value(index, index))
            else:
                self.S_dense[index, index] = 0.0

            index+=1



        if self.topK == False:
            print("Return S matrix to python caller")

            if self.sparse_weights:
                return self.S_sparse.get_scipy_csr(TopK = False)
            else:
                return np.array(self.S_dense)


        else :
            print("Return S matrix to python caller")

            if self.sparse_weights:
                return self.S_sparse.get_scipy_csr(TopK=self.topK)
            else:
                return similarityMatrixTopK(np.array(self.S_dense.T), k=self.topK, forceSparseOutput=True, inplace=True).T




    cdef SGD_sample sampleBatch_Cython(self):

        cdef SGD_sample sample = SGD_sample(-1,-1,-1.0)
        cdef long index
        cdef int numSeenItems

        #utente random preso fra tutti gli users esistenti.
        sample.user = rand() % self.n_users

        #inizializziamo il numero di item visti da quello specifico utente
        numSeenItems = self.URM_train_indptr[sample.user+1] - self.URM_train_indptr[sample.user]

        #indice random preso fra il numero di elementi visti dall'utente.
        index = rand() % numSeenItems

        sample.item = self.URM_train_indices[self.URM_train_indptr[sample.user] + index]
        sample.value = self.URM_train_data[self.URM_train_indptr[sample.user] + index]

        return sample

