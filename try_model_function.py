from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import logging

def try_model(args):
    features, X, out_of_time = args
    k_fold = KFold(10, shuffle=False)
    loo = LeaveOneOut()
    
    pid = os.getpid()
    features = list(features)
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
    #os.kill(pid, -9)
    
    return(avg_score)