import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------
# Start from scratch
infile = '/mnt/gcs-bucket/train.csv'
pcdfile = '/mnt/gcs-bucket/pcd_dataset.h5'
oufile = '/mnt/gcs-bucket/train_dataset.h5'
labelfile = '/mnt/gcs-bucket/labels.pickle'

# ---------------------------------------------------------

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
def sliceAll(df,groups,week,history):
    dfn = df[df.WeekNum == week] # preserve original
    for group in groups:
        print "\t label:"+group[1]
        dfn = pd.merge(dfn,slice(df,group[0],group[1],week,history),on=group[0],how='left')
    return dfn
        
# ---------------------------------------------------------

# read full original file
history_length = 5
df = read_original(infile,False)

## Create cross-frequency variables
## clients
#df_pcd = pd.read_hdf(pcdfile,'clients')
#sized_dict = [
#    ['DepotId','client_ndepot'],\
#    ['RouteId','client_nroute'],\
#    ['ProductId','client_nproduct']]      
#for i in sized_dict:
#    sized = df.groupby('ClientId').agg({i[0]:pd.Series.nunique}).reset_index()
#    sized = sized.rename(columns = {i[0]:i[1]})
#    sized[i[1]] = sized[i[1]].astype('float32')
#    df_pcd = pd.merge(df_pcd,sized,on='ClientId',how='left')
#store = pd.HDFStore(pcdfile)
#store['clients'] = df_pcd
#store.close()
#
## products
#df_pcd = pd.read_hdf(pcdfile,'products')
#sized_dict = [
#    ['DepotId','product_ndepot'],\
#    ['RouteId','product_nroute'],\
#    ['ClientId','product_nclient']]      
#for i in sized_dict:
#    sized = df.groupby('ProductId').agg({i[0]:pd.Series.nunique}).reset_index()
#    sized = sized.rename(columns = {i[0]:i[1]})
#    sized[i[1]] = sized[i[1]].astype('float32')
#    df_pcd = pd.merge(df_pcd,sized,on='ProductId',how='left')   
#store = pd.HDFStore(pcdfile)
#store['products'] = df_pcd
#store.close()
#
## depots
#df_pcd = pd.read_hdf(pcdfile,'depots')
#sized_dict = [
#    ['ClientId','depot_nclient'],\
#    ['RouteId','depot_nroute'],\
#    ['ProductId','depot_nproduct']]      
#for i in sized_dict:
#    sized = df.groupby('DepotId').agg({i[0]:pd.Series.nunique}).reset_index()
#    sized = sized.rename(columns = {i[0]:i[1]})
#    sized[i[1]] = sized[i[1]].astype('float32')
#    df_pcd = pd.merge(df_pcd,sized,on='DepotId',how='left')   
#store = pd.HDFStore(pcdfile)
#store['depots'] = df_pcd
#store.close()


# routes
route_client = df.groupby('RouteId').agg({'ClientId':pd.Series.nunique}).reset_index()
route_client = route_client.rename(columns = {'ClientId':'route_nclient'})
route_client.route_nclient = route_client.route_nclient.astype(np.int32)

# route_depot = df.groupby('RouteId').agg({'DepotId':pd.Series.nunique}).reset_index()
# route_depot = route_depot.rename(columns = {'DepotId':'route_ndepot'})
# route_depot.route_ndepot = route_depot.route_ndepot.astype(np.int32)

# route_product = df.groupby('RouteId').agg({'ProductId':pd.Series.nunique}).reset_index()
# route_product = route_product.rename(columns = {'ProductId':'route_nproduct'})
# route_product.route_nproduct = route_product.route_nproduct.astype(np.int32)


# Create historical demand for given combinations
label_dict = {}

groups = [\
    [['ProductId','ClientId','DepotId','RouteId','ChannelId'],'pcdrc'],\
    [['ProductId','ClientId'],'prodclient'],\
    [['ProductId','RouteId'],'prodruta'],\
    [['ClientId','RouteId'],'clientruta'],\
    [['DepotId','RouteId'],'depotruta'],\
    [['ProductId'],'prod'],\
    [['ClientId'],'client']
]

#Process price - one for all
price_label = 'price_by_pd'
price_basis = ['DepotId','ProductId']
meanprice = df.groupby(price_basis).agg({'Price': np.mean}).reset_index()
meanprice = meanprice.rename(columns = {'Price':price_label})
meanprice[price_label] = meanprice[price_label].astype('float32')
del df['Price']

#Process history - each week separately
for i in [0,1,2,3]:
    week = i + 6
    print "\nProcessing week: " +str (week)
    dfweek = sliceAll(df,groups,week,history_length)
    
    # merge with price
    # dfweek = pd.merge(dfweek,meanprice,on=price_basis,how='left')
   
    # Save to file
    dfweek.to_hdf(oufile,'week'+str(week),\
                 mode='a',\
                 complevel=9,\
                 complib='bzip2')
    dfweek = 0


# Write history labels
label_dict['history_labels'] = groups
label_dict['price_label'] = price_label
label_dict['price_basis'] = price_basis

# ---------------------------------------------------------
# Process one week at a time - add selected best features for product-client-depot
for i in [6,7,8,9]:
    df = pd.read_hdf(oufile,'week'+str(i))

    # ---------------------------------------------------------
    # # 2. Merge with product or refresh
    selected = ['ProductId',\
    'product_weight',\
    'product_mean_price',\
    'product_price_by_weight',\
    'product_shortname_tfidf_2591',\
    'product_ndepot',\
    'product_nroute',\
    'product_nclient']
    label_dict['product_labels'] = selected

    df.drop([c for c in df if c.startswith('product_')],axis=1,inplace=True)    
    df_products = pd.read_hdf(pcdfile,'products')
    df=pd.merge(df,df_products[selected],on='ProductId',how='left')
    df.product_ndepot = df.product_ndepot.astype(np.int32)
    df.product_nroute = df.product_nroute.astype(np.int32)
    df.product_nclient = df.product_nclient.astype(np.int32)
    df_products = 0

    # ---------------------------------------------------------
    # 3. Merge with client or refresh
    selected = ['ClientId',\
    'client_tfid_argmax',\
    'client_nproduct']
    label_dict['client_labels'] = selected

    df.drop([c for c in df if c.startswith('client_')],axis=1,inplace=True)    
    df_clients = pd.read_hdf(pcdfile,'clients')    
    df=pd.merge(df,df_clients[selected],on='ClientId',how='left')
    # df.client_ndepot = df.client_ndepot.astype(np.int32)
    # df.client_nroute = df.client_nroute.astype(np.int32)
    df.client_nproduct = df.client_nproduct.astype(np.int32)
    df_clients = 0

    # ---------------------------------------------------------
    # 4. Merge with towns or refresh
    selected = ['DepotId',\
    'depot_town',\
    'depot_state',\
    'depot_nproduct',\
    'depot_nclient',\
    'depot_nroute']
    label_dict['depot_labels'] = selected

    df.drop([c for c in df if c.startswith('depot_')],axis=1,inplace=True)    
    df_depot = pd.read_hdf(pcdfile,'depots')
    df=pd.merge(df,df_depot[selected],on='DepotId',how='left')
    df.depot_nroute = df.depot_nroute.astype(np.int32)
    df.depot_nproduct = df.depot_nproduct.astype(np.int32)
    df.depot_nclient = df.depot_nclient.astype(np.int32)
    df_depot = 0
    
    # ---------------------------------------------------------
    # 5. Merge with route
    selected = ['route_nclient']
    label_dict['route_labels'] = selected

    df.drop([c for c in df if c.startswith('route_')],axis=1,inplace=True)
    df=pd.merge(df,route_client,on='RouteId',how='left')
    # df=pd.merge(df,route_product,on='RouteId',how='left')
    # df=pd.merge(df,route_depot,on='RouteId',how='left')

    # ---------------------------------------------------------
    # 6. Merge with price by price basis
    df=pd.merge(df,meanprice,on=price_basis,how='left')

    # 7. Delete ChannelId as a feature
    del df['ChannelId']

    # Change target
    df.AdjDemand = np.log1p(df.AdjDemand)

    # Save the resulting trainset
    store = pd.HDFStore(oufile,complevel=9, complib='bzip2')
    store['week'+str(i)] = df
    store.close()

    df = 0

# save labels
with open(labelfile, 'wb') as handle:
    pickle.dump(label_dict, handle)

