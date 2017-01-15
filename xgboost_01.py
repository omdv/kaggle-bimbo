import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Performing grid search

# ---------------------------------------------------------
def rmsle(true,predict):
    logp = np.log1p(predict)
    logt = np.log1p(true)
    delta = np.sqrt(np.sum((logp-logt)**2)/logp.shape[0])
    return delta

def rmse(true,predict):
    delta = np.sqrt(np.sum((predict-true)**2)/predict.shape[0])
    return delta

def modelfit(alg, dtrain, target, predictors, \
    useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, 
            num_boost_round=alg.get_params()['n_estimators'],
            nfold=cv_folds,
            metrics='rmse', 
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors],dtrain[target],eval_metric='rmse')
    
    # Mean for comparison
    means = np.zeros(dtest[target].shape[0])
    means.fill(dtest[target].values.mean())

    #Print model report:
    print "\nModel Report:"
    print "RMSE (train): %.4g" % rmse(dtrain[target].values, alg.predict(dtrain[predictors]))
    print "RMSE (test): %.4g" % rmse(dtest[target].values, alg.predict(dtest[predictors]))
    print "RMSE (mean): %.4g" % rmse(dtest[target].values,means)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print "\nFeatures (cross-validation):"
    print feat_imp/1.e-2/feat_imp.values.sum()

    return alg
# ---------------------------------------------------------


# ---------------------------------------------------------
# Training and Testing
# df = pd.read_csv("processed/train.csv")
# df.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
#                     'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
#                     'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

# ---------------------------------------------------------
# training dataset
week7 = pd.read_hdf('processed/processed_train.h5','week7')
week8 = pd.read_hdf('processed/processed_train.h5','week8')
df_train = pd.concat([week7,week8])
week7 = 0
week8 = 0

# test dataset
df_test = pd.read_hdf('processed/processed_train.h5','week9')
trueAdjDemand = np.array(df_test.AdjDemand.values)
del df_test['AdjDemand']

target = 'AdjDemand'
predictors = [x for x in df_train.columns if x not in [target, 'WeekNum']]
# ---------------------------------------------------------

# ---------------------------------------------------------
xgb1 = XGBRegressor(
 learning_rate=0.1,
 n_estimators=512,
 max_depth=8,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 silent=True)

# Cross-validation
# xgb1 = modelfit(xgb1, df_train, target, predictors, df_test, useTrainCV=False)

# Validation on week 9
# xgb1.fit(df_train[predictors],df_train[target],eval_metric='rmse')
# pred = xgb1.predict(df_test[predictors])

# If NAN - assume zero, if lt zero assume zero
num_isnan = len(pred[np.isnan(pred)])/1.0/len(pred)*100.
num_ltzero = len(pred[pred<0])/1.0/len(pred)*100.
pred[np.isnan(pred)] = pred.mean()
pred[pred<0] = pred.mean()
print "nan: {:6.2f}, lt zero: {:6.2f}".\
        format(num_isnan,num_ltzero)

# Feature importances
feat_imp = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
print "\nFeatures (full set training):"
print feat_imp/1.e-2/feat_imp.values.sum()

# Score on week 9
print "RMSLE on week 9: %.4g" % rmse(trueAdjDemand,pred)