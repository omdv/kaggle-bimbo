import pandas as pd
import numpy as np
import re
#import xgboost as xgb

from nltk.collocations import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
#from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# ---------------------------------------------------------

def rmse(true,predict):
    delta = np.sqrt(np.sum((predict-true)**2)/predict.shape[0])
    return delta

# function to tokenize product names
def words_to_stems(sen,remove_modifiers=False):
    stemmer = SnowballStemmer("spanish")
    filtered_words = []
    stems = []
    
    sen = re.sub(u'\(.*\)','',sen)
    sen = re.sub(u'(\d+[Kg|ml|pct|p|g])','',sen)
    if remove_modifiers:    
        sen = re.sub(u'Super|Hot|Mini|Div|Tira','',sen)
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sen)
    
    # filter out non-alpha words
    for word in words:
       if re.search('^[a-zA-Z]', word):
           filtered_words.append(word)
    
    # stemming
    for w in filtered_words:
        stems.append(stemmer.stem(w))
           
    return ' '.join(stems)

def get_perk(val):
    res = 'None'
#    matches = re.findall(u'([A-Z]{2,})\s[A-Z]{2,}\s\d+$',val)
    matches = re.findall(u'([A-Z]{2,})\s[A-Z]{2,}\s\d+$',val)
    try:
        res = matches
    except:
        pass
    return res


#def modelfit(alg, dtrain, target, predictors, dtest,\
#    useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#    
#    if useTrainCV:
#        xgb_param = alg.get_xgb_params()
#        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#        cvresult = xgb.cv(xgb_param, xgtrain, 
#            num_boost_round=alg.get_params()['n_estimators'],
#            nfold=cv_folds,
#            metrics='rmse', 
#            early_stopping_rounds=early_stopping_rounds,
#            verbose_eval=True)
#        alg.set_params(n_estimators=cvresult.shape[0])
#    
#    #Fit the algorithm on the data
#    alg.fit(dtrain[predictors],dtrain[target],eval_metric='rmse')
#    
#    # Mean for comparison
#    means = np.zeros(dtest[target].shape[0])
#    means.fill(dtest[target].values.mean())
#
#    #Print model report:
#    print "\nModel Report:"
#    print "RMSE (train): %.4g" % rmse(dtrain[target].values, alg.predict(dtrain[predictors]))
#    print "RMSE (test): %.4g" % rmse(dtest[target].values, alg.predict(dtest[predictors]))
#    print "RMSE (mean): %.4g" % rmse(dtest[target].values,means)
#
#    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#    print "\nFeatures (cross-validation):"
#    print feat_imp/1.e-2/feat_imp.values.sum()
#
#    return alg

# ---------------------------------------------------------
# # read train

#df_full = pd.read_csv("input/train.csv")
#df_full.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
#                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
#                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
#df_full.AdjDemand = np.log1p(df_full.AdjDemand)

# read pickled product data
# df_product = pd.read_hdf('processed/processed_data.h5','products')

# # Or continue from previous point - ran by R
df_product = pd.read_csv("processed/processed_products.csv")
del df_product['Unnamed: 0']
df_product = df_product[df_product.ProductId != 0]
# ---------------------------------------------------------

## Get mean demand values by product from df_full
#by_mean = df_full.groupby('ProductId').agg({'AdjDemand':np.mean}).reset_index()
#by_mean = by_mean.rename(columns = {'AdjDemand':'product_mean_demand'})
#df_product = pd.merge(df_product,by_mean,on=['ProductId'],how='left')
#
## Get product price
#df_full['product_price'] = df_full.SalesPesosWeek/df_full.SalesUnitsWeek
#by_price = df_full.groupby('ProductId').agg({'product_price':np.mean}).reset_index()
#df_product = pd.merge(df_product,by_price,on=['ProductId'],how='left')

## New product flag
#df_product['product_had_demand'] = True
#df_product['product_had_price'] = True
#df_product.loc[df_product.product_mean_demand.isnull(),'product_had_demand'] = False
#df_product.loc[df_product.product_price.isnull(),'product_had_price'] = False

# # Process short name
df_product['product_stems'] =\
    df_product.apply(lambda row: words_to_stems(row.product_shortname),axis=1)

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(df_product.product_stems)

cls = KMeans(n_clusters = 2500)
km = cls.fit_predict(X)
df_product['product_name_cluster_max'] = km

cls = KMeans(n_clusters = 120)
km = cls.fit_predict(X)
df_product['product_name_cluster_120'] = km

cls = KMeans(n_clusters = 30)
km = cls.fit_predict(X)
df_product['product_name_cluster_30'] = km

# Process product type, which is name without modifiers
df_product['product_type'] =\
    df_product.apply(lambda row:\
    words_to_stems(row.product_shortname,True),axis=1)

# Product perk
#df_product['product_perk'] = df_product.apply(lambda row:\
#    get_perk(row.product_name),axis=1)

# # Clustering by stems





#from nltk.corpus import stopwords
#stops = stopwords.words("spanish")
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("spanish")
#producto  =  pd.read_csv("producto_tabla.csv")
#name = producto['NombreProducto'].str.extract('^((\D|\d+(pct|in))+)', expand=False)[0]
#name = name.map(lambda x: " ".join([i for i in str(x).lower().split() if i not in stops]))
#producto['stems'] = (name.map(lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()])))
#unit = producto['NombreProducto'].str.extract('(\d+)(Kg|kg|g|ml)', expand=True)
#producto['weight'] = (unit[0].astype('float')*unit[1].map({'Kg':1000, 'kg':1000, 'g':1, 'ml':1})).fillna(0.0).astype('int32')
#producto['pieces'] =  producto['NombreProducto'].str.extract('(\d+)(p|a|pq|Reb) ', expand=False)[0].fillna(1.0).astype('int32')
#perk_brand = producto['NombreProducto'].str.extract('\s((\D|CR\d)+)\s\d+$', expand=False)[0].fillna('IDENTIFICADO').str.split(' ')
#producto['perks'] = perk_brand.apply(lambda x: x[0:-1])
#producto['brand'] = perk_brand.apply(lambda x: x[-1])


# ---------------------------------------------------------
# K-mean grid search
# km_x = []
# km_y = []
# for i in [5,10,20,30,40,50,60,70,80,90,100,110]:
#     cls = KMeans(n_clusters=i)
#     cls.fit(X)
#     km_x.append(i)
#     km_y.append(cls.score(X))

# ---------------------------------------------------------
# XGBOOST part
# ---------------------------------------------------------

# # Remove unnecessary features
# df_fct = df_product
# df_fct = df_fct.drop(['Product_name','Product_shortname','Product_stems'],axis=1)
# [df_fct['Product_brand_fct'],product_brand_fct]=pd.factorize(df_fct.Product_brand)
# [df_fct['Product_type_fct'],product_type_fct]=pd.factorize(df_fct.Product_type)

# df_fct = df_fct.drop(['Product_brand','Product_type'],axis=1)

# # ---------------------------------------------------------
# # Predicting missing demand first without price
# df_train = df_fct[df_fct.AdjDemandMean.notnull()]
# df_to_predict = df_fct[df_fct.AdjDemandMean.isnull()]

# df_train.fillna(0)
# d_train, d_test = cross_validation.train_test_split(df_train,
#     test_size=0.2,random_state=42)

# # Predict Demand without price
# target = 'AdjDemandMean'
# predictors = [x for x in df_train.columns if x not in [target,'Product_price']]

# xgb1 = XGBRegressor(
#  learning_rate=0.05,
#  n_estimators=180,
#  max_depth=9,
#  min_child_weight=1,
#  gamma=0.1,
#  subsample=0.8,
#  colsample_bytree=0.6,
#  objective='reg:linear',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27,
#  silent=True)

# # Cross-validation for best estimators
# xgb1 = modelfit(xgb1, d_train, target, predictors, d_test, useTrainCV=True)

# # Fit full set
# xgb1.fit(df_train[predictors],df_train[target],eval_metric='rmse')

# # Predict adjusted demand
# prediction = xgb1.predict(df_to_predict[predictors])
# df_product.loc[df_product.AdjDemandMean.isnull(),'AdjDemandMean'] = prediction
# # ---------------------------------------------------------

# # ---------------------------------------------------------
# # Now predict product price with demand
# df_train = df_fct[df_fct.Product_price.notnull()]
# df_to_predict = df_fct[df_fct.Product_price.isnull()]

# df_train.fillna(0)
# df_train.Product_price = np.log1p(df_train.Product_price)
# d_train, d_test = cross_validation.train_test_split(df_train,
#     test_size=0.2,random_state=42)

# # Predict price with demand
# target = 'Product_price'
# predictors = [x for x in df_train.columns if x not in [target]]

# xgb1 = XGBRegressor(
#  learning_rate=0.05,
#  n_estimators=300,
#  max_depth=9,
#  min_child_weight=1,
#  gamma=0.1,
#  subsample=0.8,
#  colsample_bytree=0.6,
#  objective='reg:linear',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27,
#  silent=True)

# # Cross-validation for best estimators
# xgb1 = modelfit(xgb1, d_train, target, predictors, d_test, useTrainCV=True)

# # Fit full set
# xgb1.fit(df_train[predictors],df_train[target],eval_metric='rmse')

# # Predict adjusted demand
# prediction = xgb1.predict(df_to_predict[predictors])
# df_product.loc[df_product.Product_price.isnull(),'Product_price'] = prediction
# # ---------------------------------------------------------



# ---------------------------------------------------------
# Grid Search
# param_test1 = {
#  'max_depth':range(2,12,1),
#  'min_child_weight':range(1,6,1)
# }

# gsearch1 = GridSearchCV(
#     estimator = xgb1, 
#     param_grid = param_test1,
#     n_jobs=2,
#     iid=False,
#     cv=5,
#     fit_params={'eval_metric':'rmse'},
#     scoring = 'mean_squared_error',
#     verbose = 1)
# gsearch1.fit(d_train[predictors],d_train[target])
# print '\nGrid Scores:'
# print gsearch1.grid_scores_
# print '\nBest Params:'
# print gsearch1.best_params_

# param_test2 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }

# gsearch2 = GridSearchCV(
#     estimator = xgb1, 
#     param_grid = param_test2,
#     n_jobs=2,
#     iid=False,
#     cv=5,
#     fit_params={'eval_metric':'rmse'},
#     scoring = 'mean_squared_error',
#     verbose = 1)
# gsearch2.fit(d_train[predictors],d_train[target])
# print '\nGrid Scores:'
# print gsearch2.grid_scores_
# print '\nBest Params:'
# print gsearch2.best_params_

# param_test3 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }

# gsearch3 = GridSearchCV(
#     estimator = xgb1, 
#     param_grid = param_test3,
#     n_jobs=2,
#     iid=False,
#     cv=5,
#     fit_params={'eval_metric':'rmse'},
#     scoring = 'mean_squared_error',
#     verbose = 1)
# gsearch3.fit(d_train[predictors],d_train[target])
# print '\nGrid Scores:'
# print gsearch3.grid_scores_
# print '\nBest Params:'
# print gsearch3.best_params_

# ---------------------------------------------------------

# df_product.columns = ['ProductId', 'Product_name', 'Product_shortname', 'Product_brand',
#        'Product_weight', 'Product_pieces', 'Product_volume', 'Product_type', 'Product_wpp',
#        'Product_has_choco', 'Product_has_vanilla', 'Product_has_multigrain', 'Product_has_promotion',
#        'Product_AdjDemandMean', 'Product_price', 'Product_stems', 'Product_cluster',
#        'Product_cluster_max', 'Product_cluster_120', 'Product_is_new']

# df_product = df_product[['ProductId', 'Product_name', 'Product_shortname', 'Product_brand',
#        'Product_weight', 'Product_pieces', 'Product_volume', 'Product_type', 'Product_wpp',
#        'Product_has_choco', 'Product_has_vanilla', 'Product_has_multigrain', 'Product_has_promotion',
#        'Product_AdjDemandMean', 'Product_price', 'Product_stems',
#        'Product_cluster_max', 'Product_cluster_120', 'Product_is_new']]

# store = pd.HDFStore('processed/processed_data.h5',complevel=9, complib='bzip2')
# store['products'] = df_product
# store.close()

# Output
# df_product.set_index('ProductId').to_csv('processed/processed_products.csv')