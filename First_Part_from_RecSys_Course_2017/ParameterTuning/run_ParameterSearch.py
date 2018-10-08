#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""


from ParameterTuning.GridSearch import GridSearch
from ParameterTuning.BayesianSearch import BayesianSearch

from data.BookCrossing.BookCrossingReader import BookCrossingReader
from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from data.TheMoviesDataset.TheMoviesDatasetReader import TheMoviesDatasetReader
from data.XingChallenge2016.XingChallenge2016Reader import XingChallenge2016Reader

from data.DataSplitter import DataSplitter_ColdItems_ColdValidation


import numpy as np
import scipy.sparse as sps
import pickle


from functools import partial



def evaluation_function_cold_parameter_search_content(recommender, URM_validation, parameter_dictionary, filter_for_validation):

    return recommender.evaluateRecommendations(URM_validation, filterCustomItems=filter_for_validation, at=5, mode="sequential", exclude_seen=True)




def runParameterSearch_Content(URM_train, URM_validation, ICM_name = None, dataSplitter = None, filter_for_validation = None, logFilePath ="result_experiments/"):

    from KNN.item_knn_CBF import ItemKNNCBFRecommender
    from ParameterTuning.AbstractClassSearch import DictionaryKeys



    evaluation_function_cold_parameter_search_content_partial = partial(evaluation_function_cold_parameter_search_content,
                                                                        filter_for_validation = filter_for_validation)





    #for ICM_name in dataReader_class.AVAILABLE_ICM:

    ICM_dict = dataSplitter.get_split_for_specific_ICM(ICM_name)

    ICM_train = ICM_dict[ICM_name + "_train"]
    ICM_validation = ICM_dict[ICM_name + "_validation"]
    ICM_test = ICM_dict[ICM_name + "_test"]

    # recommender_class = ItemKNNCBFRecommender
    # parameterSearch = BayesianSearch(recommender_class, URM_validation, evaluation_function=evaluation_function_cold_parameter_search_content_partial)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [300]#[50, 100, 150, 200]
    # hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    # hyperparamethers_range_dictionary["similarity"] = ['cosine', 'jaccard']
    # hyperparamethers_range_dictionary["normalize"] = [True, False]
    # hyperparamethers_range_dictionary["feature_weighting"] = ["none"]
    #
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_" + ICM_name
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_validation, URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelPoolSize = 1)

    #parameterSearch.evaluate_on_test(URM_test)



    ##########################################################################################################



    recommender_class = ItemKNNCBFRecommender
    parameterSearch = BayesianSearch(recommender_class, URM_validation, evaluation_function=evaluation_function_cold_parameter_search_content_partial)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = ['cosine', 'jaccard']
    hyperparamethers_range_dictionary["normalize"] = [True, False]
    hyperparamethers_range_dictionary["feature_weighting"] = ["BM25"]

    output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_" + ICM_name + "_BM25"

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_validation, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



    best_parameters = parameterSearch.search(recommenderDictionary, output_root_path= output_root_path, parallelPoolSize = 8)

    parameterSearch.evaluate_on_test(URM_test)


    ##########################################################################################################



    recommender_class = ItemKNNCBFRecommender
    parameterSearch = BayesianSearch(recommender_class, URM_validation, evaluation_function=evaluation_function_cold_parameter_search_content_partial)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = ['cosine', 'jaccard']
    hyperparamethers_range_dictionary["normalize"] = [True, False]
    hyperparamethers_range_dictionary["feature_weighting"] = ["TF-IDF"]

    output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_" + ICM_name + "_TF-IDF"

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_validation, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



    best_parameters = parameterSearch.search(recommenderDictionary, output_root_path= output_root_path, parallelPoolSize = 8)

    parameterSearch.evaluate_on_test(URM_test)




def runParameterSearch_Collaborative(URM_train, URM_validation, ICM_name = None, dataSplitter = None, logFilePath ="result_experiments/"):

    from KNN.item_knn_CBF import ItemKNNCBFRecommender
    from KNN.item_knn_CF import ItemKNNCFRecommender
    from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD
    from SLIM_ElasticNet.SLIM_ElasticNet import SLIM_ElasticNet
    from SLIM_ElasticNet.Cython.SLIM_Structure_Cython import SLIM_Structure_Cython
    from GraphBased.P3alpha import P3alphaRecommender
    from GraphBased.RP3beta import RP3betaRecommender



    from ParameterTuning.AbstractClassSearch import DictionaryKeys

   ##########################################################################################################
    #
    # recommender_class = UserKNNCFRecommender
    # parameterSearch = GridSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    # hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    # hyperparamethers_range_dictionary["similarity"] = ['cosine', 'pearson', 'adjusted', 'jaccard']
    # hyperparamethers_range_dictionary["normalize"] = [True, False]
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_GridSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 8)
    #
    # parameterSearch.evaluate_on_test(URM_test)


    ##########################################################################################################
    #
    # recommender_class = ItemKNNCFRecommender
    #
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [100, 200, 500]
    # hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    # hyperparamethers_range_dictionary["similarity"] = ['cosine', 'pearson', 'adjusted', 'jaccard']
    # hyperparamethers_range_dictionary["normalize"] = [True, False]
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_BayesianSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 6)
    #
    # parameterSearch.evaluate_on_test(URM_test)



    # ##########################################################################################################
    #
    #
    # recommender_class = MultiThreadSLIM_RMSE
    # parameterSearch = GridSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100]
    # hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
    # hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]
    #
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_GridSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelize=False)
    #
    # parameterSearch.evaluate_on_test(URM_test)


   ##########################################################################################################
    #
    # recommender_class = P3alphaRecommender
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    # hyperparamethers_range_dictionary["alpha"] = list(np.arange(0.1, 2.1, 0.2))
    # hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]
    #
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_BayesianSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # parameterSearch.evaluate_on_test(URM_test)

    ##########################################################################################################
    #
    # recommender_class = RP3betaRecommender
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    # hyperparamethers_range_dictionary["alpha"] = list(np.arange(0.1, 1.7, 0.2))
    # hyperparamethers_range_dictionary["beta"] = list(np.arange(0.1, 1.7, 0.2))
    # hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]
    #
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_BayesianSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 6)
    #
    # parameterSearch.evaluate_on_test(URM_test)

    ##########################################################################################################
    #
    # recommender_class = FunkSVD
    # parameterSearch = GridSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30]
    # hyperparamethers_range_dictionary["epochs"] = [5, 10, 20]
    # hyperparamethers_range_dictionary["reg"] = [1e-2, 1e-3, 1e-4, 1e-5]
    # hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_GridSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 8)
    #
    # parameterSearch.evaluate_on_test(URM_test)
    #
    # ##########################################################################################################
    #
    # recommender_class = MF_BPR_Cython
    # parameterSearch = GridSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30]
    # hyperparamethers_range_dictionary["epochs"] = [5, 10, 20]
    # hyperparamethers_range_dictionary["batch_size"] = [1]
    # hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_GridSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 8)
    #
    # parameterSearch.evaluate_on_test(URM_test)





    #########################################################################################################
    #
    # recommender_class = SLIM_BPR_Cython
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [100, 200, 500]
    # hyperparamethers_range_dictionary["sgd_mode"] = ["adam"]
    # hyperparamethers_range_dictionary["lambda_i"] = [0.0, 1e-3, 1e-6, 1e-9]
    # hyperparamethers_range_dictionary["lambda_j"] = [0.0, 1e-3, 1e-6, 1e-9]
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_BayesianSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':True, 'positive_threshold':4,
    #                                                                    "URM_validation": URM_validation.copy()},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True, "lower_validatons_allowed":10},
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 1)
    #
    # parameterSearch.evaluate_on_test(URM_test)
















import os




URM_train = dataSplitter.get_URM_train()
URM_validation = dataSplitter.get_URM_validation()
URM_test = dataSplitter.get_URM_test()

URM_train.data[URM_train.data<=positive_threshold_train] = 0.0
URM_train.eliminate_zeros()

URM_validation.data[URM_validation.data<=positive_threshold] = 0.0
URM_validation.eliminate_zeros()

URM_test.data[URM_test.data<=positive_threshold] = 0.0
URM_test.eliminate_zeros()

test_items = dataSplitter.get_test_items()
validation_items = dataSplitter.get_validation_items()
train_items = dataSplitter.get_train_items()

filter_for_validation = np.union1d(test_items, train_items)



runParameterSearch_Content(URM_train, URM_validation, ICM_name = ICM_name, dataSplitter = dataSplitter, filter_for_validation = filter_for_validation, logFilePath=logFilePath)

