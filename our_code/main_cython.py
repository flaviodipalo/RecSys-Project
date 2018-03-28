from SLIM_RMSE_Cython_Epoch import SLIM_RMSE_Cython_Epoch
from data.movielens_1m.Movielens1MReader import Movielens1MReader
from Base.metrics import roc_auc, precision, recall, map, ndcg, rr
from Base.Recommender_utils import check_matrix, areURMequals, removeTopPop
import numpy as np
import time

data_reader = Movielens1MReader(0.8)
URM_train = data_reader.URM_train
URM_test = data_reader.URM_test

recommender = SLIM_RMSE_Cython_Epoch( URM_train, 1e-1, 5, 1e-2, 500)

#recommender.evaluate(URM_test)
