from SLIM_RMSE_Cython_Epoch import SLIM_RMSE_Cython_Epoch
from SLIM_RMSE_Cython import SLIM_RMSE_Cython
from data.movielens_1m.Movielens1MReader import Movielens1MReader
from data.movielens_10m.Movielens10MReader import Movielens10MReader

from ParameterTuning.ParameterTuning import BayesianSearch
from ParameterTuning.ParameterTuning.AbstractClassSearch import DictionaryKeys

import numpy as np
import time

print('Loading Data...')
data_reader = Movielens1MReader(0.8)

#data_reader = Movielens10MReader(0.8)
URM_train = data_reader.URM_train
URM_test = data_reader.URM_test
print('Data Loaded !')

def run_recommender():
    #cython epoch only version
    recommender = SLIM_RMSE_Cython(URM_train = URM_train, URM_validation = URM_test)
    recommender.fit()

def run_recommender_optimization():
    recommender_class = SLIM_RMSE_Cython
    parameterSearch = BayesianSearch.BayesianSearch(recommender_class,URM_test)
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [50, 100]
    hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
    hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]

    #logFilePath = 'logs/'
    #logFile = open(logFilePath + 'SLIM_RMSE_Cython' + "_GridSearch.txt", "a")

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    best_parameters = parameterSearch.search(recommenderDictionary,output_root_path='logs/')

    parameterSearch.evaluate_on_test(URM_test)

run_recommender()
#run_recommender_optimization()
