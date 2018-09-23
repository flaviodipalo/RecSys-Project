#import sys
#sys.path.append('/home/alexbacce/.local/lib/python3.6/site-packages')

import os

from MatrixFactorization.Cython import MatrixFactorization_Cython

from data.movielens_1m.Movielens1MReader import Movielens1MReader
from data.movielens_10m.Movielens10MReader import Movielens10MReader

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_Cython as mfc
from ParameterTuning.BayesianSearch import BayesianSearch as bs
from ParameterTuning.AbstractClassSearch import DictionaryKeys,EvaluatorWrapper

#Let's import evaluator
from Base.Evaluation.Evaluator import SequentialEvaluator

def run_recommender():
    #cython epoch only version
    print('Loading Data...')

    data_reader = Movielens10MReader(train_validation_split = [0.6, 0.2, 0.2],delete_popular = False)
    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    URM_validation = data_reader.URM_test

    print('Data Loaded !')
    recommender = mfc(URM_train=URM_train,URM_validation=URM_test,algorithm = "FUNK_SVD")
    recommender.fit(epochs= 15,stop_on_validation=True,validation_every_n=1,normalized_algorithm= True,sgd_mode = "adam")

def run_recommender_optimization(normalized=False, popular=False):
    print('Loading Data...')
    #TODO: fix the_movielens1M code here the data we are using are for the 1M but with the 10M code.
    data_reader = Movielens10MReader(train_validation_split = [0.6, 0.2, 0.2],delete_popular = popular)

    URM_train = data_reader.URM_train
    URM_test = data_reader.URM_test
    URM_validation = data_reader.URM_test
    print('Data Loaded !')
    ##in this part we reshape all the matrices in order to set all to the maxiumum dimention one.

    print(URM_train.nnz)
    print(URM_test.nnz)
    print(URM_validation.nnz)

    file_path = 'Norm_='+str(normalized)+'_delete_popular='+str(popular)


    evaluator_validation_earlystopping = SequentialEvaluator(URM_validation,cutoff_list= [5], exclude_seen = False)
    evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[5], exclude_seen=False)

    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    recommender_class = mfc
    parameterSearch = bs(recommender_class = recommender_class, evaluator_validation= evaluator_validation)


    hyperparamethers_range_dictionary = {}

    hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]
    hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
    hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
    hyperparamethers_range_dictionary["user_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
    #TODO: the thing is we have positive_reg = 0.0, negative_reg = 0.0,
    print("In main I see: ")
    print(repr(DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS))
    print(repr(DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS))

    '''
    recommenderDictionary = {
                              DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'URM_train':URM_train,'URM_validation':URM_test,'algorithm':'FUNK_SVD'},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: {"stop_on_validation":True,"validation_every_n":1,"normalized_algorithm":normalized},
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    '''

    recommenderDictionary = {
                              DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [],
                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'URM_train':URM_train,'algorithm':'FUNK_SVD'},
                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                              DictionaryKeys.FIT_KEYWORD_ARGS: {"stop_on_validation":True,"validation_every_n":1,"normalized_algorithm":normalized,    "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validatons_allowed": 10},
                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    parameterSearch.search(recommenderDictionary, output_root_path='new'+file_path)
    #parameterSearch.evaluate_on_test(URM_validation)

#run_recommender()
run_recommender_optimization(normalized= True)

