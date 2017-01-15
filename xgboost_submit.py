import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

history_length = 5
case_name = '/mnt/gcs-bucket/200_55feat_2016-08-23'
train_dataset = '/mnt/gcs-bucket/train_dataset.h5'
pcd_dataset = '/mnt/gcs-bucket/pcd_dataset.h5'
labels_file = '/mnt/gcs-bucket/labels.pickle'
test_csv = '/mnt/gcs-bucket/test.csv'
train_csv = '/mnt/gcs-bucket/train.csv'

# ---------------------------------------------------------
def rmse(true,predict):
    delta = np.sqrt(np.sum((predict-true)**2)/predict.shape[0])
    return delta

# slice for one combination
def slice(df,basis,label,week,history):
    hist = df.groupby(['WeekNum']+basis).agg({'AdjDemand': np.mean}).reset_index()
    pivoted = hist.pivot_table(index=basis,columns='WeekNum',values='AdjDemand').reset_index()
    labels = []
    for i in range(history):
        newlabel = '_'.join(['hist',label,str(i+1)])
        pivoted=pivoted.rename(columns = {week-i-1:newlabel})
        if newlabel in pivoted:
            labels.append(newlabel)
        else:
            pivoted[newlabel] = np.nan
            labels.append(newlabel)

    pivoted = pivoted[basis+labels]
    pivoted[basis] = pivoted[basis].astype(np.int32)
    pivoted[labels] = pivoted[labels].astype('float32')
    return pivoted

# slice all for a give combos of basis and labels
def sliceAll(df,groups,week,history=4):
    dfn = df[df.WeekNum == week] # preserve original
    for group in groups:
        print "\t label:"+group[1]
        dfn = pd.merge(dfn,slice(df,group[0],group[1],week,history),on=group[0],how='left')
    return dfn

def read_original(filename, partial = False):
    if partial:
        n = 10000
        nlines = sum(1 for l in open(filename))
        skip_idx = [x for x in range(1, nlines) if x % n != 0]
        df = pd.read_csv(filename,skiprows = skip_idx)
    else:
        df = pd.read_csv(filename)
    # re-assign columns
    df.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                     'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                     'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
    df.SalesPesosWeek = df.SalesPesosWeek.astype('float32')
    df['Price'] = df.SalesPesosWeek / df.SalesUnitsWeek
    features = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                     'ProductId','AdjDemand','Price']
    basis = ['DepotId', 'ChannelId', 'RouteId', 'ClientId','ProductId']
    df = df[features]
    df.AdjDemand = df.AdjDemand.astype('float32')
    df.WeekNum = df.WeekNum.astype(np.int32)
    df[basis] = df[basis].astype(np.int32)
    return df

# ---------------------------------------------------------

bst = xgb.Booster()
bst.load_model(case_name+'.model')

# ---------------------------------------------------------
# Prepare submission
df_submit = pd.read_csv(test_csv)
df_submit.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']
basis = ['DepotId', 'ChannelId', 'RouteId', 'ClientId','ProductId']

# ---------------------------------------------------------
# load predictors
predictors = pd.Series.from_csv(case_name+'_predictors.csv')

# read labels
with open(labels_file, 'rb') as handle:
  labels_dict = pickle.load(handle)

history_labels = labels_dict['history_labels']
product_labels = labels_dict['product_labels']
client_labels = labels_dict['client_labels']
depot_labels = labels_dict['depot_labels']
# price_label = labels_dict['price_label']
# price_basis = labels_dict['price_basis']

#save labels to case labels file
with open(case_name+'_labels.pickle', 'wb') as handle:
    pickle.dump(labels_dict, handle)

# ---------------------------------------------------------
# Predict week 10
df = read_original(train_csv,False)

#Process price - one for all
price_label = 'price_by_pd'
price_basis = ['DepotId','ProductId']
meanprice = df.groupby(price_basis).agg({'Price': np.mean}).reset_index()
meanprice = meanprice.rename(columns = {'Price':price_label})
meanprice[price_label] = meanprice[price_label].astype('float32')
del df['Price']

# calculate all route features
route_client = df.groupby('RouteId').agg({'ClientId':pd.Series.nunique}).reset_index()
route_client = route_client.rename(columns = {'ClientId':'route_nclient'})
route_client.route_nclient = route_client.route_nclient.astype(np.int32)

route_depot = df.groupby('RouteId').agg({'DepotId':pd.Series.nunique}).reset_index()
route_depot = route_depot.rename(columns = {'DepotId':'route_ndepot'})
route_depot.route_ndepot = route_depot.route_ndepot.astype(np.int32)

route_product = df.groupby('RouteId').agg({'ProductId':pd.Series.nunique}).reset_index()
route_product = route_product.rename(columns = {'ProductId':'route_nproduct'})
route_product.route_nproduct = route_product.route_nproduct.astype(np.int32)

# merge with week 10
df_week10 = df.append(df_submit[df_submit.WeekNum == 10]).reset_index()
del df_week10['index']
df = 0

# 1. Add histories
df_week10_predict = sliceAll(df_week10,history_labels,10,history=history_length)

# 2. Merge with product
df_products = pd.read_hdf(pcd_dataset,'products')
df_week10_predict =pd.merge(df_week10_predict,df_products[product_labels],on='ProductId',how='left')
df_products = 0

# 3. Merge with client
df_clients = pd.read_hdf(pcd_dataset,'clients')
df_week10_predict =pd.merge(df_week10_predict,df_clients[client_labels],on='ClientId',how='left')
df_clients = 0

# 4. Merge with depot
df_depot = pd.read_hdf(pcd_dataset,'depots')
df_week10_predict =pd.merge(df_week10_predict,df_depot[depot_labels],on='DepotId',how='left')
df_depot = 0

# 5. Merge with prices
df_week10_predict =pd.merge(df_week10_predict,meanprice,on=price_basis,how='left')

# 6. Merge with route
df_week10_predict=pd.merge(df_week10_predict,route_client,on='RouteId',how='left')
# df_week10_predict=pd.merge(df_week10_predict,route_product,on='RouteId',how='left')
# df_week10_predict=pd.merge(df_week10_predict,route_depot,on='RouteId',how='left')

print "\nShape of week 10: {}\n".format(df_week10_predict.shape)

# predictions
pred = bst.predict(xgb.DMatrix(df_week10_predict[predictors].values))
df_week10.loc[df_week10.WeekNum == 10, 'AdjDemand'] = np.expm1(pred)
df_week10_predict = 0
pred = 0

# ---------------------------------------------------------
# Predict week 11
df_week11 = df_week10.append(df_submit[df_submit.WeekNum == 11]).reset_index()
# df_week11.loc[df_week11.WeekNum == 10,'AdjDemand'] = np.nan
del df_week11['index']

# 1. Add histories
df_week11_predict = sliceAll(df_week11,history_labels,11,history=history_length)

# 2. Merge with product
df_products = pd.read_hdf(pcd_dataset,'products')
df_week11_predict =pd.merge(df_week11_predict,df_products[product_labels],on='ProductId',how='left')
df_products = 0
# if product_mean_demand:
# del df_week11_predict['product_mean_demand']
# mean_product = df_week11.groupby('ProductId').agg({'AdjDemand': np.nanmean}).reset_index()
# mean_product = mean_product.rename(columns = {'AdjDemand':'product_mean_demand'})
# df_week11_predict = pd.merge(df_week11_predict,mean_product,on='ProductId',how='left')


# 3. Merge with client
df_clients = pd.read_hdf(pcd_dataset,'clients')
df_week11_predict =pd.merge(df_week11_predict,df_clients[client_labels],on='ClientId',how='left')
df_clients = 0
# if client_mean_demand:
# del df_week11_predict['client_mean_demand']
# mean_client = df_week11.groupby('ClientId').agg({'AdjDemand': np.nanmean}).reset_index()
# mean_client = mean_client.rename(columns = {'AdjDemand':'client_mean_demand'})
# df_week11_predict = pd.merge(df_week11_predict,mean_client,on='ClientId',how='left')



# 4. Merge with depot
df_depot = pd.read_hdf(pcd_dataset,'depots')
df_week11_predict =pd.merge(df_week11_predict ,df_depot[depot_labels],on='DepotId',how='left')
df_depot = 0
# if depot_mean_demand:
# del df_week11_predict['depot_mean_demand']
# mean_depot = df_week11.groupby('DepotId').agg({'AdjDemand': np.nanmean}).reset_index()
# mean_depot = mean_depot.rename(columns = {'AdjDemand':'depot_mean_demand'})
# df_week11_predict = pd.merge(df_week11_predict,mean_depot,on='DepotId',how='left')

# 5. Merge with prices
df_week11_predict =pd.merge(df_week11_predict,meanprice,on=price_basis,how='left')

# 6. Merge with route
df_week11_predict=pd.merge(df_week11_predict,route_client,on='RouteId',how='left')
# df_week11_predict=pd.merge(df_week11_predict,route_product,on='RouteId',how='left')
# df_week11_predict=pd.merge(df_week11_predict,route_depot,on='RouteId',how='left')


print "\nShape of week 11: {}\n".format(df_week11_predict.shape)

pred = bst.predict(xgb.DMatrix(df_week11_predict[predictors].values))
df_week11.loc[df_week11.WeekNum == 11, 'AdjDemand'] = np.expm1(pred)
df_week11_predict = 0
pred = 0

# ---------------------------------------------------------
# Save submission file
df_week10 = df_week10[df_week10.WeekNum == 10]
df_week10.id = df_week10.id.astype(np.int32)
df_week10 = df_week10[['id','AdjDemand']]

df_week11 = df_week11[df_week11.WeekNum == 11]
df_week11.id = df_week11.id.astype(np.int32)
df_week11 = df_week11[['id','AdjDemand']]

df_submit = pd.concat([df_week10,df_week11]).set_index('id')
df_submit.columns = ['Demanda_uni_equil']

df_submit.to_csv(case_name+'_submit.csv')