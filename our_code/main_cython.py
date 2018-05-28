from SLIM_RMSE_Cython_Epoch import SLIM_RMSE_Cython_Epoch
from SLIM_RMSE_Cython import SLIM_RMSE_Cython
from data.movielens_1m.Movielens1MReader import Movielens1MReader
from data.movielens_10m.Movielens10MReader import Movielens10MReader
import argparse

from ParameterTuning.ParameterTuning import BayesianSearch
from ParameterTuning.ParameterTuning.AbstractClassSearch import DictionaryKeys

#ssh -i /Users/flaviodipalo/Downloads/recsys-project.pem ubuntu@131.175.21.230
parser = argparse.ArgumentParser()
parser.add_argument("epoch", type=int)

args = parser.parse_args()
epoch = args.epoch


import numpy as np
import time

print('Loading Data...')

#data_reader = Movielens10MReader(train_test_split = 0.8, delete_popular = True)
data_reader = Movielens1MReader(train_test_split = 0.8, delete_popular=True)

#if you need also a validation set.
#data_reader = Movielens1MReader(train_test_split = 0.6,train_validation_split=0.5)

#data_reader = Movielens10MReader(0.8)
URM_train = data_reader.URM_train
URM_test = data_reader.URM_test
print('Data Loaded !')

def run_recommender(epoch):
    #cython epoch only version
    recommender = SLIM_RMSE_Cython(URM_train = URM_train, URM_validation = URM_test)
    recommender.fit(epoch)
    recommender.evaluate(URM_test)

def run_recommender_optimization():
    recommender_class = SLIM_RMSE_Cython
    parameterSearch = BayesianSearch.BayesianSearch(recommender_class,URM_test)
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [50, 100]
    hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
    hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    parameterSearch.search(recommenderDictionary,output_root_path='logs/')

    parameterSearch.evaluate_on_test(URM_test)

#run_recommender(epoch)
run_recommender_optimization()
