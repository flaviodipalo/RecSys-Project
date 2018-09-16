#import sys
#sys.path.append('/home/alexbacce/.local/lib/python3.6/site-packages')

import os

from MatrixFactorization.Cython import MatrixFactorization_Cython

from data.movielens_1m.Movielens1MReader import Movielens1MReader
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython as mfc
from ParameterTuning.BayesianSearch import BayesianSearch as bs
from ParameterTuning.AbstractClassSearch import DictionaryKeys

def run_recommender():
    #cython epoch only version
    print('Loading Data...')

    #data_reader = Movielens1MReader(train_test_split=0.6,train_validation_split = 0.5, delete_popular=False)
    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    URM_validation = data_reader.URM_validation

    print('Data Loaded !')
    recommender = mfc(URM_train=URM_train,URM_validation=URM_test,algorithm = "FUNK_SVD")
    recommender.fit(epochs= 15,stop_on_validation=True,validation_every_n=1,normalized_algorithm= True)


def run_recommender_optimization(normalized=False, popular=False):
    print('Loading Data...')
    #TODO: fix this part in order to get correct results
    data_reader = Movielens1MReader(train_test_split=0.6,train_validation_split=0.5, delete_popular=popular)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    #TODO:pay attention here
    URM_validation = data_reader.URM_test
    print('Data Loaded !')

    #TODO: run
    print(URM_train.nnz)
    print(URM_test.nnz)
    print(URM_validation.nnz)

    #recommender = mfc(URM_train=URM_train,URM_validation=URM_test,algorithm = "FUNK_SVD")
    #recommender.fit(epochs= 15,stop_on_validation=True,validation_every_n=1,normalized_algorithm= True)

    file_path = 'Norm_='+str(normalized)+'_delete_popular='+str(popular)

    recommender_class = mfc
    parameterSearch = bs(recommender_class, URM_validation)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["learning_rate"] = [0.01]

    print("In main I see: ")
    print(repr(DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS))
    print(repr(DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS))
# {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train,URM_validation],

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train,URM_test],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    parameterSearch.search(recommenderDictionary, output_root_path='new'+file_path)
    parameterSearch.evaluate_on_test(URM_test)

run_recommender_optimization()