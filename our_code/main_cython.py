from cython_epoch_with_class import CythonEpoch
from data.movielens_1m.Movielens1MReader import Movielens1MReader
from Base.metrics import roc_auc, precision, recall, map, ndcg, rr
from Base.Recommender_utils import check_matrix, areURMequals, removeTopPop
import numpy as np
import time

data_reader = Movielens1MReader(0.8)
URM_train = data_reader.URM_train
URM_test = data_reader.URM_test

recommender = CythonEpoch(URM_train)
recommender.fit(learning_rate = 1e-1, gamma=5, beta=1e-2, iterations=500, threshold=5)
#recommender.evaluate(URM_test)

def get_user_relevant_items(self, user_id):
    return self.URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

def evaluateRecommendationsSequential(self, usersToEvaluate):

    start_time = time.time()

    roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n_eval = 0

    for test_user in usersToEvaluate:

        # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

        # Being the URM CSR, the indices are the non-zero column indexes
        relevant_items = self.get_user_relevant_items(test_user)

        n_eval += 1

        recommended_items = recommender.recommend(test_user)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY
        roc_auc_ += roc_auc(is_relevant)
        precision_ += precision(is_relevant)
        recall_ += recall(is_relevant, relevant_items)
        map_ += map(is_relevant, relevant_items)
        mrr_ += rr(is_relevant)

        if n_eval % 10000 == 0 or n_eval == len(usersToEvaluate) - 1:
            print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                n_eval,
                100.0 * float(n_eval + 1) / len(usersToEvaluate),
                time.time() - start_time,
                float(n_eval) / (time.time() - start_time)))

    if (n_eval > 0):
        roc_auc_ /= n_eval
        precision_ /= n_eval
        recall_ /= n_eval
        map_ /= n_eval
        mrr_ /= n_eval
        ndcg_ /= n_eval

    else:
        print("WARNING: No users had a sufficient number of relevant items")

    results_run = {}

    results_run["AUC"] = roc_auc_
    results_run["precision"] = precision_
    results_run["recall"] = recall_
    results_run["map"] = map_
    results_run["NDCG"] = ndcg_
    results_run["MRR"] = mrr_

    return (results_run)

#recommender.fit(URM_train=URM_train, learning_rate=1e-1, lasso_term=5, ridge_term=1e-2 , iterations_in=500 , threshold_in=5 )
#