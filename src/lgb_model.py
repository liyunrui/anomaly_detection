import time
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import KFold, StratifiedKFold
import multiprocessing   
import lightgbm as lgb
from sklearn.metrics import f1_score


def lgb_f1_score(y_true, y_pred):
    """evaluation metric"""
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True


# lgb model
def kfold_lightgbm(df_train, df_test, num_folds, args, logger, stratified = False, seed = int(time.time())):
    """
    LightGBM GBDT with KFold or Stratified KFold
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import multiprocessing   
    import lightgbm as lgb
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    #train_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in ["label"]]
    # k-fold
    if args.TEST_NULL_HYPO:
        # shuffling our label for feature selection
        df_train['label'] = df_train['label'].copy().sample(frac=1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['label'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['label'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['label'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        if args.TEST_NULL_HYPO:
            clf = lgb.LGBMClassifier(
                n_jobs = int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
                n_estimators=10000,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )
        else:
            clf = lgb.LGBMClassifier(
                n_jobs = int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
                n_estimators=50000,
                learning_rate=0.001, # 0.02
                num_leaves=int(args.NUM_LEAVES),
                colsample_bytree=args.COLSAMPLE_BYTREE,
                subsample=args.SUBSAMPLE,
                subsample_freq=args.SUBSAMPLE_FREQ,
                max_depth=args.MAX_DEPTH,
                reg_alpha=args.REG_ALPHA,
                reg_lambda=args.REG_LAMBDA,
                min_split_gain=args.MIN_SPLIT_GAIN,
                min_child_weight=args.MIN_CHILD_WEIGHT,
                max_bin=args.MAX_BIN,
                silent=-1,
                verbose=-1,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )

        if args.TEST_NULL_HYPO:
            clf.fit(train_x, 
                    train_y, 
                    eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= "auc", 
                    verbose= True, 
                    early_stopping_rounds= 100, 
                    categorical_feature='auto') # early_stopping_rounds= 200
        else:
            clf.fit(train_x, 
                    train_y, 
                    eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= lgb_f1_score, 
                    verbose= False, 
                    early_stopping_rounds= 5000, 
                    categorical_feature='auto') # early_stopping_rounds= 200
        # probabilty belong to class1(fraud)
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logger.info('Fold %2d val f1-score : %.6f' % (n_fold + 1, lgb_f1_score(valid_y, oof_preds[valid_idx])[1]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    logger.info('---------------------------------------\n')
    over_folds_val_score = lgb_f1_score(df_train['label'], oof_preds)[1]
    logger.info('Over-folds val f1-score %.6f\n---------------------------------------' % over_folds_val_score)
    # Write submission file and plot feature importance

    if args.ensemble:
        df_test.loc[:,'label'] = sub_preds
        df_test[['id', 'label']].to_csv("../result/lgb.csv", index= False)

    df_test.loc[:,'pred'] = np.round(sub_preds)
    df_test.loc[:,'prob'] = sub_preds
    #df_test.to_csv(args.result_path, index= False)
    df_test[['id', "prob",'pred',"label"]].to_csv(args.result_path, index= False)
    
    return feature_importance_df, over_folds_val_score

