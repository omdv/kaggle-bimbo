import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
# from sklearn import cross_validation, metrics   #Additional scklearn functions
# from sklearn.grid_search import GridSearchCV   #Performing grid search

# ---------------------------------------------------------
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
xgb1 = XGBRegressor(
 learning_rate=0.1,
 n_estimators=50,
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

# ---------------------------------------------------------
# training dataset
week7 = pd.read_hdf('processed/processed_train.h5','week7')
week8 = pd.read_hdf('processed/processed_train.h5','week8')
df_train = pd.concat([week7,week8])
week7 = 0
week8 = 0

print "\nTraining Dataset:"
print df_train.shape

target = 'AdjDemand'
predictors = [x for x in df_train.columns if x not in [target, 'WeekNum']]
# ---------------------------------------------------------

# # Train on weeks 7-8
xgb1.fit(df_train[predictors],df_train[target],eval_metric='rmse')
xgb1._Booster.save_model('50_weeks78.model')

# # predict test dataset
df_test = pd.read_hdf('processed/processed_train.h5','week9')
pred = xgb1.predict(df_test[predictors])

print "\nPredicting Week 9:"
print df_test.shape

#Print model report:
print "\nModel Report:"
print "RMSE (train): %.4g" % rmse(df_train[target].values, xgb1.predict(df_train[predictors]))
print "RMSE (test): %.4g" % rmse(df_test[target].values, xgb1.predict(df_test[predictors]))

feat_imp = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
print "\nFeatures (cross-validation):"
print feat_imp/1.e-2/feat_imp.values.sum()

# ---------------------------------------------------------
# # Train on full set
# df = pd.concat([df_test,df_train])
# df_test = 0
# df_train = 0

# del df['product_has_integral']
# del df['product_has_multigrain']
# del df['product_has_vanilla']
# # del df['product_has_choco']
# # del df['product_volume']
# predictors = [x for x in df.columns if x not in [target, 'WeekNum']]

# print "\nTraining on Full set:"
# print df.shape

# xgb1.fit(df[predictors],df[target],eval_metric='rmse')
# xgb1._Booster.save_model('512_weeks789.model')

# #Print model report:
# print "\nModel Report:"
# print "RMSE (train): %.4g" % rmse(df[target].values, xgb1.predict(df[predictors]))

# feat_imp = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
# print "\nFeatures (full set):"
# print feat_imp/1.e-2/feat_imp.values.sum()
