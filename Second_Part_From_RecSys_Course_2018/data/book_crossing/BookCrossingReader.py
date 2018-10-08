import numpy as np
import scipy.sparse as sps
import os
import random
import sys
import pandas as pd


class BookCrossingReader(object):
    #TODO: aggiungere validation option.
    def __init__(self, train_test_split=None, train_validation_split=None, delete_popular=None, top_popular_threshold=0.33):
        '''
        :param train_test_split: is the percentage of the training set
        '''


        dir = os.path.dirname(__file__)
        filename = dir+"/BX-Book-Ratings.csv"
        from numpy import genfromtxt
        fileHandle = pd.read_csv(filename, sep=";", encoding="ISO-8859-1")
        #fileHandle = open(filename, "r")

        rows, cols, vals = [], [], []
        numCells = 0
        fileHandle['ISBN'], levels = pd.factorize(fileHandle['ISBN'])
        fileHandle['User-ID'], levels = pd.factorize(fileHandle['User-ID'])
        print(fileHandle.iloc[0:100])

        #fileHandle['User'], levels = pd.factorize(fileHandle['ISBN'] )

        #print(fileHandle)

        #These arrays are sorted by user
        self.users = np.array(fileHandle['User-ID']).astype(int)
        self.movies = np.array(fileHandle['ISBN']).astype(int)
        self.ratings = np.array(fileHandle['Book-Rating']).astype(float)

        if delete_popular:
            unique, counts = np.unique(self.movies, return_counts=True)
            dictionary = dict(zip(unique, counts))
            sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1])
            cutting_index = round(len(sorted_dictionary)*(1-top_popular_threshold))
            least_popular_item = [x[0] for x in sorted_dictionary[:cutting_index]]

            popular_mask = []
            for item in self.movies:

                if item in least_popular_item:
                    popular_mask.append(True)
                else:
                    popular_mask.append(False)

            self.movies = self.movies[popular_mask]
            self.users = self.users[popular_mask]
            self.ratings = self.ratings[popular_mask]

        self.unique_movies = np.sort(np.unique(self.movies)).astype(int)
        self.unique_users = np.sort(np.unique(self.users))
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
        self.URM_all = self.URM_all.tocoo()

        num_interactions = self.URM_all.nnz

        if train_validation_split is not None:
            split = np.random.choice([1, 2, 3], num_interactions, p=train_validation_split)

            train_mask = split == 1

            test_mask = split == 2

            validation_mask = split == 3


            max_user_id = max(self.users) +1
            max_item_id = max(self.movies) +1

            self.URM_validation = sps.coo_matrix((self.URM_all.data[validation_mask],
                                                  (self.URM_all.row[validation_mask], self.URM_all.col[validation_mask])),shape=(max_user_id,max_item_id))
            self.URM_validation = self.URM_validation.tocsr()

        else:
            raise Exception("One between train_test_split and train_validation_split must be valid")

        #mask = np.random.choice([True, False], len(self.unique_users), p=[0.3, 0.7])

        self.URM_test = sps.csr_matrix((self.ratings[test_mask], (self.users[test_mask], self.movies[test_mask])))
        print(self.URM_test.nnz)
        print(self.URM_test.shape[0])
        self.URM_test = self.URM_test[0:1000, :]
        print(self.URM_test.nnz)

        self.URM_train = sps.csr_matrix((self.ratings[train_mask], (self.users[train_mask], self.movies[train_mask])))
        self.URM_train = self.URM_train[0:1000, :]

        #print(num_interactions, self.URM_test.nnz, self.URM_train.nnz, self.URM_validation.nnz)

#dataset = BookCrossingReader(0.8,0.9)
