#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Base.Recommender import Recommender
from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
from Base.Recommender_utils import similarityMatrixTopK

import subprocess
import os, sys, time

import numpy as np

def default_validation_function(self):

    return self.evaluateRecommendations(self.URM_validation)

class SLIM_RMSE_Cython(Similarity_Matrix_Recommender, Recommender):
    RECOMMENDER_NAME = "SLIM_RMSE_Recommender"

    def __init__(self, URM_train, URM_validation=None, recompile_cython=False):

        super(SLIM_RMSE_Cython, self).__init__()

        self.URM_train = URM_train.copy()
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        self.URM_mask = self.URM_train.copy()

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")
#
    def fit(self,learning_rate = 1e-2, l1_penalty=0, l2_penalty=0,topK = 300, logFile='SLIM_RMSE_training.log',validation_every_n = 1,validation_function=None,
            stop_on_validation=True, lower_validatons_allowed=5, validation_metric="map",epochs=13,similarity_matrix_normalized=False,URM_validation=None):
        print(self.URM_validation is None)
        if  self.URM_validation is None and URM_validation is not None :
            self.URM_validation = URM_validation.copy()

        print('fit has started',stop_on_validation,validation_every_n,l1_penalty,l2_penalty,topK,similarity_matrix_normalized)

        self.sparse_weights = False
        self.train_with_sparse_weights = False
        self.epochs = epochs
        self.batch_size = 1

        # Import compiled module
        from SLIM_RMSE_Cython_Epoch import SLIM_RMSE_Cython_Epoch

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        self.cythonEpoch = SLIM_RMSE_Cython_Epoch(URM_train=self.URM_train,learning_rate = learning_rate, gamma=l1_penalty, beta=l2_penalty, iterations=1, gradient_option="adagrad",similarity_matrix_normalized=similarity_matrix_normalized)

        if (topK != False and topK < 1):
            raise ValueError(
                "TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(
                    topK))
        self.topK = topK

        self.logFile = logFile

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        if validation_function is None:
            validation_function = default_validation_function

        self.learning_rate = learning_rate

        start_time = time.time()

        best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()
        self.epochs_best = 0
        currentEpoch = 0

        while currentEpoch < epochs and not convergence:

            if self.batch_size > 0:
                print('Running Epoch Number:',currentEpoch)
                self.cythonEpoch.epochIteration_Cython()
            else:
                print("No batch not available")

            # Determine whether a validaton step is required
            if self.URM_validation is not None and (currentEpoch + 1) % self.validation_every_n == 0:

                print("SLIM_RMSE_Cython: Validation begins...")

                self.get_S_incremental_and_set_W()
                results_run = validation_function(self)

                print("SLIM_RMSE_Cython: {}".format(results_run))

                # Update the D_best and V_best
                # If validation is required, check whether result is better
                if stop_on_validation:

                    current_metric_value = results_run[validation_metric]

                    if best_validation_metric is None or best_validation_metric < current_metric_value:

                        best_validation_metric = current_metric_value

                        self.S_best = self.S_incremental.copy()
                        self.epochs_best = currentEpoch + 1

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validatons_allowed:
                        convergence = True
                        print(
                            "SLIM_RMSE_Cython: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                                currentEpoch + 1, validation_metric, self.epochs_best, best_validation_metric,
                                (time.time() - start_time) / 60))
                        ##TODO:Added THIS
                        #self.W = self.S_best.copy()

            # If no validation required, always keep the latest
            if not stop_on_validation:
                self.S_best = self.S_incremental.copy()

            print("SLIM_RMSE_Cython: Epoch {} of {}. Elapsed time {:.2f} min".format(
                currentEpoch + 1, self.epochs, (time.time() - start_time) / 60))

            currentEpoch += 1
        self.W = self.S_best.copy()
        #self.get_S_incremental_and_set_W()

        sys.stdout.flush()

    #written by us
    def evaluate(self,URM_test_external,logFile):
        print("SLIM_RMSE_Cython: Validation begins...")

        self.get_S_incremental_and_set_W()
        results_run = self.evaluateRecommendations(URM_test_external)

        print("SLIM_RMSE_Cython: {}".format(results_run))

        if (logFile != None):
            logFile.write("Test Set Results {}\n".format(results_run))


    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            # logFile.write("Weights: {}\n".format(str(list(self.weights))))
            logFile.flush()

    '''
    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/SLIM_BPR/Cython"
        # fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]

            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True,
                                                 cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass

        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx
    '''
    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
        else:
            if self.sparse_weights:
                self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.topK)
            else:
                self.W = self.S_incremental


