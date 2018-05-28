from SLIM_RMSE_Cython_Epoch import SLIM_RMSE_Cython_Epoch
from SLIM_RMSE_Cython import SLIM_RMSE_Cython
from data.movielens_1m.Movielens1MReader import Movielens1MReader
from data.movielens_10m.Movielens10MReader import Movielens10MReader
import argparse

from ParameterTuning import BayesianSearch
from ParameterTuning.AbstractClassSearch import DictionaryKeys

#ssh -i /Users/flaviodipalo/Downloads/recsys-project.pem ubuntu@131.175.21.230
parser = argparse.ArgumentParser()
parser.add_argument("normalized", type=bool)
parser.add_argument("popular", type=bool)


args = parser.parse_args()
normalized = args.normalized
popular = args.popular

import numpy as np
import time



def run_recommender(epoch):
    #cython epoch only version
    recommender = SLIM_RMSE_Cython(URM_train = URM_train, URM_validation = URM_test)

    recommender.fit(epochs=epoch,similarity_matrix_normalized=False)

def run_recommender_optimization(normalized = False, popular = False):
    print('Loading Data...')
    data_reader = Movielens1MReader(train_test_split=0.8, delete_popular=popular)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    print('Data Loaded !')
    #the file path that will print the solution for each configuration file
    file_path = 'Norm_='+str(normalized)+'_delete_popular='+str(popular)

    recommender_class = SLIM_RMSE_Cython
    parameterSearch = BayesianSearch.BayesianSearch(recommender_class,URM_test)
#
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [50, 100]
    hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
    hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]
    hyperparamethers_range_dictionary["similarity_matrix_normalized"] = [normalized]


    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    parameterSearch.search(recommenderDictionary,output_root_path='logs/'+file_path)

    #the next function is used to evaluate with the test set while training with validation
    #parameterSearch.evaluate_on_test(URM_test)

#run_recommender(epoch)
run_recommender_optimization(normalized,popular)

