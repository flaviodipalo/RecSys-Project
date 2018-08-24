import sys
sys.path.append('/home/alexbacce/.local/lib/python3.6/site-packages')

from SLIM_RMSE_Cython import SLIM_RMSE_Cython
from data.movielens_1m.Movielens1MReader import Movielens1MReader
from data.book_crossing.BookCrossingReader import BookCrossingReader
from data.movielens_10m.Movielens10MReader import Movielens10MReader
import argparse

from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD

def run_recommender():
    #cython epoch only version
    print('Loading Data...')
    data_reader = Movielens1MReader(train_test_split=0.8, delete_popular=False)
    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test

    print('Data Loaded !')

    recommender = FunkSVD(URM_train)
    recommender.fit(epochs= 15)
    print(recommender.evaluateRecommendations(URM_test))

run_recommender()