from SLIM_RMSE_Cython_Epoch import SLIM_RMSE_Cython_Epoch
from SLIM_RMSE_Cython import SLIM_RMSE_Cython
from data.movielens_1m.Movielens1MReader import Movielens1MReader

import numpy as np
import time

data_reader = Movielens1MReader(0.8)
URM_train = data_reader.URM_train
URM_test = data_reader.URM_test

#cython epoch only version
#recommender = SLIM_RMSE_Cython_Epoch( URM_train, 1e-1, 5, 1e-2, 500)
#recommender.evaluate(URM_test)

recommender = SLIM_RMSE_Cython(URM_train = URM_train)
recommender.fit()