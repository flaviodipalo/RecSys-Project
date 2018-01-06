import numpy as np
import scipy.sparse as sps
import os

class Movielens1MReader(object):
    def __init__(self,train_test_split):
        '''
        :param train_test_split: is the percentage of the training set
        '''

        dir = os.path.dirname(__file__)
        filename = dir+"/ratings.dat"

        data = np.loadtxt(filename, delimiter="::")
        users = np.array(data[:,0])
        movies = np.array(data[:,1])
        ratings = np.array(data[:,2])

        self.URM_all = sps.csr_matrix((ratings, (users, movies)), dtype=np.float32)

        numInteractions = self.URM_all.nnz
        train_mask = np.random.choice([True,False], numInteractions, p=[train_test_split, 1-train_test_split])
        test_mask = np.logical_not(train_mask)

        URM_train = sps.coo_matrix((ratings[train_mask], (users[train_mask], movies[train_mask])))
        URM_test = sps.coo_matrix((ratings[test_mask], (users[test_mask], movies[test_mask])))

        self.URM_train = URM_train.tocsr()
        self.URM_test = URM_test.tocsr()
