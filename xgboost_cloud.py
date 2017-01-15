import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import time

infile = '/mnt/gcs-bucket/train_dataset.h5'

# ---------------------------------------------------------
def rmse(true,predict):
    delta = np.sqrt(np.sum((predict-true)**2)/predict.shape[0])
    return delta

def modelfit(alg, dtrain, dtest, target, predictors, \
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

# Train on full set
df_train = pd.concat([\
            pd.read_hdf(infile,'week6'),\
            pd.read_hdf(infile,'week7'),\
            pd.read_hdf(infile,'week8')\
            ])
df_test = pd.read_hdf(infile,'week9')

# create a small validation set - 0.3%
msk = np.random.rand(df_test.shape[0]) < 0.003
df_train = pd.concat([df_train,df_test[~msk]])
df_test = df_test[msk]

##plug to convert frequency features to np.int32
#for i in ['depot_nclient','depot_nroute','depot_nproduct',\
#            'product_ndepot','product_nroute','product_nclient',\
#            'client_ndepot','client_nroute','client_nproduct',\
#            'route_nclient','route_ndepot','route_nproduct']:
#    df_train[i] = df_train[i].astype('int')
#    df_train[i] = df_train[i].astype(np.int32)
#    df_test[i] = df_test[i].astype('int')
#    df_test[i] = df_test[i].astype(np.int32)

print "\nTraining on Full set of size: {}".format(df_train.shape)

target = 'AdjDemand'
predictors = pd.Series([x for x in df_train.columns if x not in [target, 'WeekNum']])

# Create DMatrix
df_train = xgb.DMatrix(df_train[predictors.values].values,label=df_train[target].values)
df_test = xgb.DMatrix(df_test[predictors.values].values,label=df_test[target].values)

print "\nDMatrix initiated..."

# Validation set
evallist  = [(df_train,'train'),(df_test,'eval')]

num_round = 200
num_features = predictors.shape[0]

params = {
    'booster':'gbtree',
    'nthread':8,
    'max_depth':10,
    'eta':0.1,
    'min_child_weight':1,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'tree_method':'exact',
    'silent':'true'
}

# TODO - annealed learning rate
# Train and save booster
bst = xgb.train( 
        params,\
        df_train,\
        num_boost_round=num_round,
        early_stopping_rounds=10,\
        evals=evallist,\
        verbose_eval=1)

# Feature importances - merge with predictors
fi_dict = bst.get_fscore()
fi_pred = {}
for key in fi_dict:
    num = int(key.replace('f',''))
    nkey = predictors[num]
    fi_pred[nkey] = fi_dict[key]

feat_imp = pd.Series(fi_pred).sort_values(ascending=False)
feat_imp = feat_imp/1.e-2/feat_imp.values.sum()

print "\nFeatures (full set):"
print feat_imp

# Saving model and feature importances
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
modelfile = '/mnt/gcs-bucket/{}_{}feat_{}'.format(num_round,num_features,timestamp)

bst.save_model(modelfile+'.model')
feat_imp.to_csv(modelfile+'_features.csv')
predictors.to_csv(modelfile+'_predictors.csv')

print "\nCase saved to file: {}".format(modelfile)

