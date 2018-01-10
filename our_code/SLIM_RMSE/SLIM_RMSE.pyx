import scipy.sparse as sps
import numpy as np
cimport numpy as np

class SLIM_RMSE():
    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.URM_mask = self.URM_train.copy()
        #S = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)

    def SLIM_RMSE_epoch(self,matrix):
        cdef double[:,:] S = np.random.rand(self.n_items,1)
        cdef float beta = 0.1
        cdef int gamma = 10
        cdef int j = 0
        cdef int i = 0
        cdef int[:,:] URM_train = matrix
        print(S.size)
        print(S)

        #inizializzare a random positivi la matrice S
        function = 1/2*(np.linalg.norm(URM_train[:,j]-np.dot(URM_train,S[:,j]),2))^2 + (beta/2)*(np.linalg.norm(S[:,j]))^2 + gamma
        gradient_w_i_j = (URM_train[:,j] - np.dot(URM_train,S[:,j]))*URM_train[i,j] + beta*S[i,j] + gamma

        print(function)

