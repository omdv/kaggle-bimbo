import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Performing grid search
from sklearn.ensemble import RandomForestRegressor

# to_predict
df_to_predict = pd.read_csv("processed/missing_pr.csv")
df_train = pd.read_csv("processed/by_pr.csv",nrows=5000)
df_train.AdjDemand = np.log1p(df_train.AdjDemand)

# Define target and predictors
predictors = ['ProductId', 'RouteId']
# predictors = [x for x in df_train.columns if x not in [target]]
target = 'AdjDemand'

# Combine two dataframes


# ---------------------------------------------------------

def modelfit(alg, df, target, predictors, useTrainCV=True, 
            early_stopping_rounds=5, cv_folds=5):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(df[predictors].values, label=df[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, 
            num_boost_round=alg.get_params()['n_estimators'], 
            nfold=cv_folds,
            metrics='rmse',
            early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print cvresult.shape[0]

    # create dummy variables and update predictors
    df_bin = pd.get_dummies(df,columns=['ProductId','RouteId'])
    predictors_bin = [x for x in df_bin.columns if x not in [target]]

    train, test = cross_validation.train_test_split(
        df_bin,test_size=0.33,random_state=42)
    
    #Fit the algorithm on the data
    alg.fit(train[predictors_bin],train[target],eval_metric='rmse')
    alg._Booster.save_model('pcd_001.model')
        
    #Predict training set:
    predictions = alg.predict(test[predictors_bin])

    # Predict by mean
    global_mean = np.empty(test.shape[0])
    global_mean.fill(train.AdjDemand.mean())
    product_mean = df.iloc[train.index].groupby('ProductId').agg({'AdjDemand':np.mean})
    mean_predict = pd.merge(df.iloc[test.index][predictors],product_mean.reset_index(),on=['ProductId'],how='left')
    mean_predict = mean_predict.AdjDemand.fillna(train.AdjDemand.mean())
        
    #Print model report:
    print "\nModel Report"
    print "RMSE (xgboost): %.4g" % metrics.mean_squared_error(test[target].values,\
        predictions)
    print "RMSE (product-mean): %.4g" % metrics.mean_squared_error(test[target].values,\
        mean_predict.values)
    print "RMSE (global-mean): %.4g" % metrics.mean_squared_error(test[target].values,\
        global_mean)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print "\nFeatures:"
    print feat_imp
    return alg

# ---------------------------------------------------------

xgb1 = XGBRegressor(
 learning_rate=0.1,
 n_estimators=128,
 max_depth=10,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 silent=True)

xgb1 = modelfit(xgb1, df_train, target, predictors, useTrainCV=False)


# predict missing
df_to_predict_bin = pd.get_dummies(df_to_predict,columns=['ProductId','RouteId'])
predictors = [x for x in df_to_predict_bin.columns if x not in [target]]

pred = xgb1.predict(df_to_predict_bin[predictors])
df_to_predict['AdjDemand'] = np.expm1(pred)

df_to_predict.set_index('ProductId').to_csv('processed/predicted_pr.csv')

