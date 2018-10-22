import pandas as pd
import numpy as np
import os
from datetime import datetime
import seaborn as sns
import math
import logging
import random

def main():
    random.seed(42)

    base_modelagem = pd.read_csv("base_delta_cross_L5.csv", index_col = 0)

    filtrada = base_modelagem.iloc[len(base_modelagem)-1063:len(base_modelagem)]
    print("Tamanho da base filtrada:", len(filtrada))

    print("Proporção da variável resposta:")
    print(filtrada.fl_home_win.value_counts()/len(filtrada))

    base_modelo = filtrada.iloc[0:(len(filtrada) - 100)]
    out_of_time = filtrada.iloc[(len(filtrada) - 100):len(filtrada)]

    import sklearn.base
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import KFold, LeaveOneOut
    import time

    from mlxtend.feature_selection import SequentialFeatureSelector as sfs
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs

    X = base_modelo.drop("fl_home_win", axis=1)
    y = base_modelo.fl_home_win

    logging.basicConfig(filename='ft_selection' + str(int(time.time())) + '.log', filemode='w', format='%(asctime)s;%(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.debug("time;feature_names;avg_kfold;kfold_scores;min_kfold;max_kfold;avg_leave_one_out;holdout_score")

    k_fold = KFold(10, shuffle=False)
    loo = LeaveOneOut()

    def try_model(features):
        pid = os.getpid()
        print(features, pid)
        clf = GaussianNB()
        X_ = X[features]

        # KFold 10
        score_k_folds = []
        for train_index, test_index in k_fold.split(X_):
            test_model = GaussianNB()
            test_model.fit(X_.iloc[train_index], y.iloc[train_index])

            y_test_pred = test_model.predict(X_.iloc[test_index])
            y_test_true = y.iloc[test_index]
            score_k_folds.append(accuracy_score(y_test_true, y_test_pred))

        avg_score = np.mean(score_k_folds)
        min_score = np.min(score_k_folds)
        max_score = np.max(score_k_folds)

        # Leave One Out
        score_loo_folds = []
        for train_index, test_index in loo.split(X_):
            test_model = GaussianNB()
            test_model.fit(X_.iloc[train_index], y.iloc[train_index])

            y_test_pred = test_model.predict(X_.iloc[test_index])
            y_test_true = y.iloc[test_index]
            score_loo_folds.append(accuracy_score(y_test_true, y_test_pred))

        avg_loo_score = np.mean(score_loo_folds)

        # Holdout
        test_model = GaussianNB()
        test_model.fit(X_, y)

        y_pred_holdout = test_model.predict(out_of_time[features])
        y_true_holdout = out_of_time.fl_home_win
        holdout_score = accuracy_score(y_true_holdout, y_pred_holdout)

        # Loga os resultados
        # logging.debug("feature_names;avg_kfold;kfold_scores;min_kfold;max_kfold;avg_leave_one_out;holdout_score")    
        logging.debug(";".join([str(features), str(avg_score), str(min_score), str(max_score), str(avg_loo_score), str(holdout_score)]))

        # Mata o processo
        os.kill(pid, -9)

    # Gera permutações
    from itertools import permutations 
    X = X[['D2_PFD_L 5', 'D2_NET_RATING_L 5', 'D2_EFG_PCT_L 5', 'C1_TO_L 5', 'C2_OPP_TOV_PCT_L 5']]
    l = list(permutations(X.columns, 4)) 
    print("Combinações a serem testadas:", len(l))

    import multiprocessing
    from multiprocessing import Pool

    p = Pool(multiprocessing.cpu_count() - 1) 
    r = p.imap(try_model, l)
    p.close()
    p.join()     

if __name__ == '__main__':
    main()
