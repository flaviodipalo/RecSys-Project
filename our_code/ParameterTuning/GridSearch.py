#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/17

@author: Maurizio Ferrari Dacrema
"""

import itertools
import multiprocessing
from functools import partial
import traceback







def writeLog(string, logFile):

    print(string)

    if logFile!=None:
        logFile.write(string)
        logFile.flush()


from ParameterTuning.AbstractClassSearch import AbstractClassSearch, DictionaryKeys

class GridSearch(AbstractClassSearch):

    def __init__(self, recommender_class, URM_validation, evaluation_function=None):

        super(GridSearch, self).__init__(recommender_class, URM_validation,  evaluation_function = evaluation_function)




    def runSingleCase(self, paramether_dictionary, dictionary, folderPath = None, namePrefix = None):

        try:

            # Create an object of the same class of the imput
            # Passing the paramether as a dictionary
            recommender = self.recommender_class(*dictionary[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                 **dictionary[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])


            print("GridSearch: Testing config: {}".format(paramether_dictionary))

            recommender.fit(*dictionary[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **dictionary[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **paramether_dictionary)


            paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(recommender, paramether_dictionary)


            if folderPath != None:
                recommender.saveModel(folderPath, namePrefix = namePrefix)


            #return recommender.evaluateRecommendations(self.URM_validation, at=5, mode="sequential")
            return self.evaluation_function(recommender, self.URM_validation, paramether_dictionary), paramether_dictionary_to_save


        except Exception as e:

            print("GridSearch: Testing config: {} - Exception {}\n".format(paramether_dictionary, str(e)))
            traceback.print_exc()

            return None



    def search(self, dictionary, metric ="map", logFile = None, parallelPoolSize = 2, parallelize = True,
               folderPath = None, namePrefix = None):

        hyperparamethers_range_dictionary = dictionary[DictionaryKeys.FIT_RANGE_KEYWORD_ARGS]

        key_list = list(hyperparamethers_range_dictionary.keys())

        # Unpack list ranges from hyperparamethers to validate onto
        # * operator allows to transform a list of objects into positional arguments
        test_cases = itertools.product(*hyperparamethers_range_dictionary.values())

        paramether_dictionary_list = []

        at_least_one_evaluation_done = False

        for current_case in test_cases:

            paramether_dictionary = {}

            for index in range(len(key_list)):

                paramether_dictionary[key_list[index]] = current_case[index]

            paramether_dictionary_list.append(paramether_dictionary)

            #results_test = self.runSingleCase(dictionary, paramether_dictionary, logFile)

            if len(paramether_dictionary_list) >= parallelPoolSize or not parallelize:

                at_least_one_evaluation_done = True
                self.evaluateBlock(dictionary, paramether_dictionary_list, metric, logFile, parallelPoolSize, parallelize)

                # Reset paramether list for next block
                paramether_dictionary_list = []


        if not at_least_one_evaluation_done:
            # Test cases are less than number of parallel threads
            at_least_one_evaluation_done = True
            self.evaluateBlock(dictionary, paramether_dictionary_list, metric, logFile, parallelPoolSize, parallelize)


        writeLog("GridSearch: Best config is: Config {}, {} value is {:.4f}\n".format(self.paramether_dictionary_best, metric, self.results_test_best[metric]), logFile)

        if folderPath != None:

            writeLog("BayesianSearch: Saving model in {}\n".format(folderPath), logFile)
            self.runSingleCase(self.paramether_dictionary_best, metric, folderPath = folderPath, namePrefix = namePrefix)


        return self.paramether_dictionary_best



    def evaluateBlock(self, dictionary, paramether_dictionary_list, metric, logFile, parallelPoolSize, parallelize):

        if parallelize:

            runSingleCase_partial = partial(self.runSingleCase,
                                            dictionary=dictionary)

            pool = multiprocessing.Pool(processes=parallelPoolSize, maxtasksperchild=1)
            resultList = pool.map(runSingleCase_partial, paramether_dictionary_list)

            pool.close()

        else:
            resultList = self.runSingleCase(paramether_dictionary_list[0], dictionary)
            resultList = [resultList]


        for results_index in range(len(resultList)):

            results_test, paramether_dictionary_test = resultList[results_index]

            if results_test!=None:

                if metric not in self.results_test_best or results_test[metric] > self.results_test_best[metric]:

                    self.results_test_best = results_test.copy()
                    self.paramether_dictionary_best = paramether_dictionary_test.copy()

                    writeLog("GridSearch: New best config found. Config {}, {} value is {:.4f}\n".format(paramether_dictionary_test, metric, self.results_test_best[metric]), logFile)

                else:
                    writeLog("GridSearch: Config is suboptimal. Config {}, {} value is {:.4f}\n".format(paramether_dictionary_test, metric, results_test[metric]), logFile)







if __name__ == '__main__':

    from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
    from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader

    dataReader = NetflixEnhancedReader()
    URM_train = dataReader.get_URM_train()
    URM_test = dataReader.get_URM_test()

    logFile = open("BPR_MF_GridSearch.txt", "a")


    gridSearch = GridSearch(MF_BPR_Cython, None, URM_test, None)


    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["num_factors"] = list(range(1, 51, 5))
    hyperparamethers_range_dictionary["epochs"] = list(range(1, 51, 10))
    hyperparamethers_range_dictionary["batch_size"] = list(range(1, 101, 50))
    hyperparamethers_range_dictionary["learning_rate"] = [1e-1, 1e-2, 1e-3, 1e-4]



    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    best_paramethers = gridSearch.search(recommenderDictionary, logFile = logFile)

    print(best_paramethers)