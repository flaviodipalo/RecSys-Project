from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
import numpy as np
from subprocess import call
import shlex
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))

if __name__ == '__main__':

    data_reader = Movielens1MReader(0.8)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    #print(URM_train)

    users = data_reader.users
    movies = data_reader.movies
    ratings = data_reader.ratings

    users_by_item = data_reader.users_by_item
    items_by_item = data_reader.items_by_item
    ratings_by_item = data_reader.ratings_by_item


    # || A - AW || per una sola riga
    # la 1 esima colonna della matrice A è data da.
    A = URM_train[:, 1]

    #passiamo a calcolare la prediction ora.
    #print(URM_train[:,:].shape[1])
    #n_movies = URM_train[:,:].shape[1]
    # we initialize the first colum of the S matrix (also called W matrix)
    #S = np.random.rand(n_movies, 1)
    #S[0] = 0
    #print(S)

    #print(len(S))

    #prediction = URM_train.dot(S)
    #frobenius norm between the prediction and the value.
    #for the first step let's immagine we want to minimize this:
    #print (np.linalg.norm(A-prediction,'fro'))

    #la stima della stessa colonna imparata è
#    recommender_list = []
    #recommender_list.append(SLIM_BPR_Cython(URM_train, sparse_weights=False))
#    recommender_list.append(SLIM_RMSE(URM_train))
    rec_object = SLIM_RMSE(URM_train)
    #add cython compiling
    #rec_object.SLIM_RMSE_epoch(URM_train)
    rec_object.SLIM_RMSE_epoch(URM_train, users, movies, ratings, users_by_item, items_by_item, ratings_by_item) #prova

#inizializziamo la W random con la diagonale a 0.

'''
    for recommender in recommender_list:

        print("Algorithm: {}".format(recommender.__class__))

        recommender.fit()

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        print("Algorithm: {}, results: {}".format(recommender.__class__, results_run))
'''
