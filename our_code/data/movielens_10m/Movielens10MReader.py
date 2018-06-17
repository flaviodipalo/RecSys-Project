import numpy as np
import scipy.sparse as sps
import os
import pandas as pd

class Movielens10MReader(object):
    # TODO: aggiungere validation option.
    def __init__(self, train_test_split=None, train_validation_split=None, delete_popular=None, top_popular_threshold = 0.33, delimiter = "::"):
        '''
        :param train_test_split: is the percentage of the training set
        '''

        dir = os.path.dirname(__file__)
        filename = dir + "/ratings.dat"


        rows, cols, vals = [], [], []
        numCells = 0

        fileHandle = open(filename, "r")

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(delimiter)

                line[-1] = line[-1].replace("\n", "")

                if not line[2] == "0" and not line[2] == "NaN":
                    rows.append(int(line[0]))
                    cols.append(int(line[1]))
                    vals.append(float(line[2]))
        # data2 = np.loadtxt(filename2, delimiter="::")

        # These arrays are sorted by user
        self.users = np.array(rows)
        self.movies = np.array(cols)
        self.ratings = np.array(vals)
        print(len(self.users))
        self.users, unique_users = pd.factorize(self.users)
        self.movies, unique_movies = pd.factorize(self.movies)

        if delete_popular:

            print("Eliminating top {}% popular items...".format(top_popular_threshold * 100))
            unique, counts = np.unique(unique_movies, return_counts=True)
            dictionary = dict(zip(unique, counts))
            sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1])
            print(len(sorted_dictionary))
            cutting_index = round(len(sorted_dictionary) * (1 - top_popular_threshold))
            least_popular_item = set([x[0] for x in sorted_dictionary[:cutting_index]])
            print(len(least_popular_item))
            popular_mask = []
            numCells = 0

            fileHandle = open(filename, "r")

            for line in fileHandle:
                numCells += 1
                if (numCells % 1000000 == 0):
                    print("Processed {} cells".format(numCells))

                if (len(line)) > 1:
                    line = line.split(delimiter)

                    line[-1] = line[-1].replace("\n", "")

                    if not line[2] == "0" and not line[2] == "NaN":
                        if int(line[1]) in least_popular_item:
                            popular_mask.append(True)
                        else:
                            popular_mask.append(False)

            self.users = self.users[popular_mask]
            self.movies = self.movies[popular_mask]
            self.ratings = self.ratings[popular_mask]

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

            self.URM_validation = sps.coo_matrix((self.URM_all.data[validation_mask],
                                                  (self.URM_all.row[validation_mask], self.URM_all.col[validation_mask])))
            self.URM_validation = self.URM_validation.tocsr()

        elif train_test_split is not None:
            train_mask = np.random.choice([True, False], num_interactions, p=[train_test_split, 1 - train_test_split])

            test_mask = np.logical_not(train_mask)

        else:
            raise Exception("One between train_test_split and train_validation_split must be valid")

        self.URM_test = sps.csr_matrix((self.ratings[test_mask], (self.users[test_mask], self.movies[test_mask])))

        self.URM_train = sps.csr_matrix((self.ratings[train_mask], (self.users[train_mask], self.movies[train_mask])))
        print(self.URM_train.shape)
        unique_movies = np.unique(self.movies)
        print(len(unique_movies), max(unique_movies))
        #self.URM_train = URM_train[0:, :]
        #self.URM_test = URM_test[0:, :]



