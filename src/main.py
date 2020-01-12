import argparse
from config import Configs
from contextlib import contextmanager
import time
import logging
from time import strftime, localtime
import pandas as pd
import numpy as np
import sys
from lgb_model import kfold_lightgbm
from sklearn.model_selection import train_test_split
# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
#log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
log_file = '../result/{}.log'.format(strftime("%y%m%d-%H%M", localtime()))
logger.addHandler(logging.FileHandler(log_file))

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}s".format(title, time.time() - t0))


def main(args):
    with timer("loading data"):
        #-------------------------
        # load dataset
        #-------------------------
        payment_df = pd.read_csv(args.payment_data_path)
        customer_df = pd.read_csv(args.customer_data_path)

        df = pd.merge(customer_df,payment_df, on = "id", how = "left")
        df.sort_values(by = ["id","update_date"], inplace = True)
        #-------------------------
        # pre-processing
        #-------------------------
        # handle categorial data
        for cat in Configs.CATEGORY:
            df[cat] = df[cat].astype('category') #.cat.codes

        # # time-related
        # payment_df.update_date = pd.to_datetime(payment_df.update_date, format='%d/%m/%Y')
        # payment_df.report_date = pd.to_datetime(payment_df.report_date, format='%d/%m/%Y')
        
    with timer("Process train/test application"):
        if args.split_data_ratio:
            n_sample_test = int(customer_df.id.nunique() * 0.1)
            test_user = customer_df.sample(frac = 1.0).id.unique()[:n_sample_test]

            df_train = df[~df.id.isin(test_user)]
            df_test = df[df.id.isin(test_user)]
            # test
            assert len(df_train)+len(df_test),"split data into train/test is wrong"
        else:
            df_train = df
            df_test = df
        

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add customer id feature"):
    #     from util import group_target_by_cols
    #     df_train, df_test = group_target_by_cols(df_train, df_test, Configs.BALANCE_AGG_RECIPE)

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Run LightGBM with kfold"):
        ITERATION = (5 if args.TEST_NULL_HYPO else 1)
        feature_importance_df = pd.DataFrame()
        over_iterations_val_auc = np.zeros(ITERATION)
        for i in range(ITERATION):
            logger.info('Iteration %i' %i)
            if args.model == "lgb":    
                iter_feat_imp, over_folds_val_auc = kfold_lightgbm(df_train, df_test, num_folds = args.NUM_FOLDS, args = args, stratified = args.STRATIFIED, seed = args.SEED, logger = logger)
            elif args.model == "linear regression":
                iter_feat_imp, over_folds_val_auc = kfold_xgb(df_train, df_test, num_folds = args.NUM_FOLDS, args = args, stratified = args.STRATIFIED, seed = args.SEED, logger = logger)
            else:
                print("Now we only support LightGBM or Xgboost model!")           
            feature_importance_df = pd.concat([feature_importance_df, iter_feat_imp], axis=0)
            over_iterations_val_auc[i] = over_folds_val_auc

        logger.info('============================================\nOver-iterations f1-score  %.6f' %over_iterations_val_auc.mean())
        #logger.info('Standard deviation %.6f\n============================================' %over_iterations_val_auc.std())
    
    if args.feature_importance_plot == True:
        from util import display_importances
        display_importances(feature_importance_df, args.model)
        
    feature_importance_df_median = feature_importance_df[["feature", "importance"]].groupby("feature").median().sort_values(by="importance", ascending=False)
    useless_features_df = feature_importance_df_median.loc[feature_importance_df_median['importance'] == 0]
    feature_importance_df_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

    if args.TEST_NULL_HYPO:
        feature_importance_df_mean.to_csv("../result/feature_importance-null_hypo.csv", index = True)
    else:
        feature_importance_df_mean.to_csv("../result/feature_importance.csv", index = True)
        useless_features_list = useless_features_df.index.tolist()
        logger.info('Useless features: \'' + '\', \''.join(useless_features_list) + '\'')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--payment_data_path', default='../data/payment_data_ratio20.csv', type=str)
    parser.add_argument('--customer_data_path', default='../data/customer_data_ratio20.csv', type=str)
    parser.add_argument('--result_path', default='../result/submission.csv', type=str)

    # lgbm parameters(needed to be filled in with best parameters eventually)
    parser.add_argument('--NUM_FOLDS', default=10, type=int, help='number of folds we split for out-of-fold validation')
    parser.add_argument('--SEED', default=1030, type=int, help='set seed for reproducibility')
    parser.add_argument('--NUM_LEAVES', default=31, type=int, help='Maximum tree leaves for base learners.')
    parser.add_argument('--CPU_USE_RATE', default=1.0, type=float, help='0~1 use how many percentanges of cpu')
    parser.add_argument('--COLSAMPLE_BYTREE', default=1.0, type=float, help = "Subsample ratio of columns when constructing each tree.")
    parser.add_argument('--SUBSAMPLE', default=1.0, type=float, help= " Subsample ratio of the training instance.")
    parser.add_argument('--SUBSAMPLE_FREQ', default=0, type=int, help='Frequence of subsample, <=0 means no enable.')
    parser.add_argument('--MAX_DEPTH', default=-1, type=int, help='Maximum tree depth for base learners, <=0 means no limit.')
    parser.add_argument('--REG_ALPHA', default=0.0, type=float, help = "L1 regularization term on weights.")
    parser.add_argument('--REG_LAMBDA', default=0.0, type=float,  help = "L2 regularization term on weights")
    parser.add_argument('--MIN_SPLIT_GAIN', default=0.0, type=float, help = "Minimum loss reduction required to make a further partition on a leaf node of the tree.")
    parser.add_argument('--MIN_CHILD_WEIGHT', default=0.001, type=float, help= "Minimum sum of instance weight (hessian) needed in a child (leaf).")
    parser.add_argument('--MAX_BIN', default=255, type=int, help='max number of bins that feature values will be bucketed in,  constraints: max_bin > 1')
    parser.add_argument('--SCALE_POS_WEIGHT', default=5.0, type=float, help = "weight of labels with positive class")
    # para
    parser.add_argument('--split_data_ratio', default=0.1, type=float, help='split dataset into train and test based on customer[0.0 to 1.0]')
    parser.add_argument('--feature_importance_plot', default=True, type=bool, help='plot feature importance')
    parser.add_argument('--feature_selection', default=False, type=bool, help='drop unused features and random features (by null hypothesis). If true, need to provide features set in list format')
    parser.add_argument('--STRATIFIED', default=True, type=bool, help='use STRATIFIED k-fold. Otherwise, use k-fold')
    parser.add_argument('--TEST_NULL_HYPO', default=False, type=bool, help='get random features by null hypothesis')
    parser.add_argument('--ensemble', default=False, type=bool, help='save testing results with predicted prob for ensemble')
    parser.add_argument('--model', default='lgb', type=str, help='lgb or linear regression')

    main(parser.parse_args())
