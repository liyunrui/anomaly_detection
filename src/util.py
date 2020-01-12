import pandas as pd
import gc
# Display/plot feature importance
def display_importances(feature_importance_df_, model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if model == "lgb":
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(32, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout
        plt.savefig('../result/lgbm_importances.png')
    elif model == "xgb":
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(32, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('Xgboost Features (avg over folds)')
        plt.tight_layout
        plt.savefig('../result/xgb_importances.png')
    else:
        print("Now we only support LightGBM or Xgboost model!")  

def group_target_by_cols(df_train, df_test, recipe):
    df = pd.concat([df_train, df_test], axis = 0)
    for m in range(len(recipe)):
        cols = recipe[m][0]
        for n in range(len(recipe[m][1])):
            target = recipe[m][1][n][0]
            method = recipe[m][1][n][1]
            name_grouped_target = method+"_"+target+'_BY_'+'_'.join(cols)
            tmp = df[cols + [target]].groupby(cols).agg(method)
            tmp = tmp.reset_index().rename(index=str, columns={target: name_grouped_target})
            df_train = df_train.merge(tmp, how='left', on=cols)
            df_test = df_test.merge(tmp, how='left', on=cols)

        # reduced memory    
        del tmp
        gc.collect()
    
    return df_train, df_test