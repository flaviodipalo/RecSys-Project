#import sys
#sys.path.append('/home/alexbacce/.local/lib/python3.6/site-packages')

import os


from data.movielens_1m.Movielens1MReader import Movielens1MReader

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython

def run_recommender():
    #cython epoch only version
    print('Loading Data...')
    data_reader = Movielens1MReader(train_test_split=0.8, delete_popular=False)
    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test

    print('Data Loaded !')
    recommender = MatrixFactorization_Cython(URM_train=URM_train,URM_validation=URM_test,algorithm = "FUNK_SVD")
    recommender.fit(epochs= 15,stop_on_validation=True,validation_every_n=1,normalized_algorithm= True)

run_recommender()