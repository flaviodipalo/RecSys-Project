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
        filename2 = dir+"/ratings_ordered_by_item.dat"

        print("Loading data...")

        data = np.loadtxt(filename, delimiter="::")
        #data2 = np.loadtxt(filename2, delimiter="::")

        #These arrays are sorted by user
        self.users = np.array(data[:,0])
        self.movies = np.array(data[:,1])
        self.ratings = np.array(data[:,2])
        '''
        #These arrays are sorted by item
        self.users_by_item = np.array(data2[:,0])
        self.items_by_item = np.array(data2[:,1])
        self.ratings_by_item = np.array(data2[:,2])

        # gli id degli users partono da 1 e sono tutti consecutivi, quindi l'unica
        # riga della URM che ha tutti 0 è la prima (riga 0) che quindi eliminiamo
        '''
        URM_all_partial = sps.csr_matrix((self.ratings, (self.users, self.movies)), dtype=np.float32)
        self.URM_all = URM_all_partial

        numInteractions = self.URM_all.nnz
        train_mask = np.random.choice([True,False], numInteractions, p=[train_test_split, 1-train_test_split])
        test_mask = np.logical_not(train_mask)

        URM_train = sps.csr_matrix((self.ratings[train_mask], (self.users[train_mask], self.movies[train_mask])))
        URM_test = sps.csr_matrix((self.ratings[test_mask], (self.users[test_mask], self.movies[test_mask])))

        self.URM_train = URM_train[0:, :]
        self.URM_test = URM_test[0:, :]
