from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
from subprocess import call
import shlex
#call(shlex.split('python3 /home/alessio/PycharmProjects/RecSys_Project/our_code/SLIM_RMSE/setup.py build_ext --inplace'))

if __name__ == '__main__':

    data_reader = Movielens1MReader(0.8)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    users = users = data_reader.users
    movies = data_reader.movies
    ratings = data_reader.ratings
    recommender_list = []
    #recommender_list.append(SLIM_BPR_Cython(URM_train, sparse_weights=False))
    #recommender_list.append(SLIM_RMSE(URM_train))
    rec_object = SLIM_RMSE(URM_train)
    #add cython compiling
    #rec_object.SLIM_RMSE_epoch(URM_train)
    rec_object.SLIM_RMSE_epoch(URM_train, users, movies, ratings) #prova

'''
    for recommender in recommender_list:

        print("Algorithm: {}".format(recommender.__class__))

        recommender.fit()

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        print("Algorithm: {}, results: {}".format(recommender.__class__, results_run))
'''
