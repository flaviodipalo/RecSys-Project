#import sys
#sys.path.append('/home/alexbacce/.local/lib/python3.6/site-packages')

import os

from MatrixFactorization.Cython import MatrixFactorization_Cython

from data.movielens_1m.Movielens1MReader import Movielens1MReader
from data.book_crossing.BookCrossingReader import BookCrossingReader
from data.epinions_dataset.EpinionsReader import EpinionsReader
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython as mfc
from ParameterTuning.BayesianSearch import BayesianSearch as bs
from ParameterTuning.AbstractClassSearch import DictionaryKeys,EvaluatorWrapper

from Base.Evaluation.Evaluator import SequentialEvaluator

from telegram_bot import TelegramBot

def run_recommender():
    #cython epoch only version
    print('Loading Data...')

    #data_reader = Movielens1MReader(train_validation_split = [0.6, 0.2, 0.2],delete_popular = False)
    data_reader = BookCrossingReader(train_validation_split = [0.6, 0.2, 0.2],delete_popular = False)
    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    URM_validation = data_reader.URM_test

    print('Data Loaded !')
    recommender = mfc(URM_train=URM_train,URM_validation=URM_test,algorithm = "FUNK_SVD")
    recommender.fit(epochs= 15,stop_on_validation=True,validation_every_n=1,normalized_algorithm= True,sgd_mode = "adam")

def run_recommender_optimization(normalized=False, popular=False):

    print('Loading Data...')
    #data_reader = Movielens1MReader(train_validation_split = [0.6, 0.2, 0.2],delete_popular = popular)
    data_reader = BookCrossingReader(train_validation_split = [0.6, 0.2, 0.2],delete_popular = False,delete_interactions=0.5)
    #data_reader = EpinionsReader(train_validation_split=[0.6, 0.2, 0.2], delete_popular=False)
    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    URM_validation = data_reader.URM_test
    print('Data Loaded !')
    ##in this part we reshape all the matrices in order to set all to the maxiumum dimention one.


    file_path = 'Norm_='+str(normalized)+'_delete_popular='+str(popular)


    evaluator_validation_earlystopping = SequentialEvaluator(URM_validation,cutoff_list= [5], exclude_seen = False)
    evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[5], exclude_seen=False)

    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    recommender_class = mfc
    parameterSearch = bs(recommender_class = recommender_class, evaluator_validation= evaluator_validation,evaluator_test = evaluator_test)

    hyperparamethers_range_dictionary = {}

    hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4]
    hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
    hyperparamethers_range_dictionary["num_factors"] = [5, 10, 20, 30, 50, 70, 90, 110]
    hyperparamethers_range_dictionary["user_reg"] = [0.0, 1e-3, 1e-6]
    hyperparamethers_range_dictionary["epochs"] = [10,20,50,100,150,300]

    recommenderDictionary = {
                              DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'URM_train':URM_train,'algorithm':'FUNK_SVD'},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: {"stop_on_validation":False,"validation_every_n":301,"normalized_algorithm":normalized,    "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validatons_allowed": 500},
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    parameterSearch.search(recommenderDictionary, output_root_path='new'+file_path)
    parameterSearch.evaluate_on_test()

run_recommender_optimization(normalized= False)
run_recommender_optimization(normalized= True)



