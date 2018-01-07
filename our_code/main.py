from data.movielens_1m.Movielens1MReader import Movielens1MReader
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE

if __name__ == '__main__':

    data_reader = Movielens1MReader(0.8)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test

    recommender_list = []
    #recommender_list.append(SLIM_BPR_Cython(URM_train, sparse_weights=False))
    #recommender_list.append(SLIM_RMSE(URM_train))
    rec_object = SLIM_RMSE(URM_train)
    #add cython compiling
    rec_object.SLIM_RMSE_epoch(URM_train)

'''
    for recommender in recommender_list:

        print("Algorithm: {}".format(recommender.__class__))

        recommender.fit()

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        print("Algorithm: {}, results: {}".format(recommender.__class__, results_run))
'''
