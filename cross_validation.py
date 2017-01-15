import pandas as pd
import numpy as np

# Training and Testing
df = pd.read_csv("input/train.csv")
df.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

df_train = df[df.WeekNum < 9]

df_test = df[df.WeekNum == 9]
trueAdjDemand = df_test.AdjDemand
df_test = df_test[['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']]

df_client = pd.read_csv("processed/processed_clients_cluster.csv")
df_client.columns = ['ClientId','ClientName','stems','ClusterId']

df_client = df_client[['ClientId','ClusterId']]
df_train = pd.merge(df_train,df_client,on=['ClientId'],how='left')
df_test = pd.merge(df_test,df_client,on=['ClientId'],how='left')

def rmsle(true,predict):
    logp = np.log1p(predict)
    logt = np.log1p(true)
    delta = np.sqrt(np.sum((logp-logt)**2)/logp.shape[0])
    return delta

# ---------------------------------------------------------

df_train.AdjDemand = np.log1p(df_train.AdjDemand)

# Product-Client-Depot
by_pcd = df_train.groupby(['ProductId','ClientId','DepotId']).agg({'AdjDemand': np.mean}).reset_index()
by_pcd.AdjDemand = np.expm1(by_pcd.AdjDemand)*0.82147681+0.45916909

# Product - Client Cluster
by_pcc = df_train.groupby(['ProductId','ClusterId']).agg({'AdjDemand': np.mean}).reset_index()
by_pcc.AdjDemand = np.expm1(by_pcc.AdjDemand)*0.86921662+0.57833923

# Product-Depot
by_pd = df_train.groupby(['ProductId','DepotId']).agg({'AdjDemand': np.mean}).reset_index()
by_pd.AdjDemand = np.expm1(by_pd.AdjDemand)*0.54+0.8

# Product-Route
by_pr = df_train.groupby(['ProductId','RouteId']).agg({'AdjDemand': np.mean}).reset_index()
by_pr.AdjDemand = np.expm1(by_pr.AdjDemand)*0.85406138+0.64441297

# Client
by_c = df_train.groupby(['ClientId']).agg({'AdjDemand': np.mean}).reset_index()
by_c.AdjDemand = np.expm1(by_c.AdjDemand)*0.097+1.0521

# Product
by_p = df_train.groupby(['ProductId']).agg({'AdjDemand': np.mean}).reset_index()
by_p.AdjDemand = np.expm1(by_p.AdjDemand)*0.44+1.

by_pmiss = pd.read_csv("processed/predicted_products.csv")
by_pmiss = by_pmiss[['ProductId','AdjDemandMean']]
by_pmiss.columns = ['ProductId','AdjDemand']

ipcd = 0
ipcc = 0
ipr = 0
ipd = 0
ic = 0
ip = 0

# ---------------------------------------------------------
df_pcd = pd.merge(df_test,by_pcd,on=['ProductId','ClientId','DepotId'],how='left')
df_pcc = pd.merge(df_test,by_pcc,on=['ProductId','ClusterId'],how='left')
df_pd = pd.merge(df_test,by_pd,on=['ProductId','DepotId'],how='left')
df_pr = pd.merge(df_test,by_pr,on=['ProductId','RouteId'],how='left')
df_c = pd.merge(df_test,by_c,on=['ClientId'],how='left')
df_p = pd.merge(df_test,by_p,on=['ProductId'],how='left')
df_pmiss = pd.merge(df_test,by_pmiss,on=['ProductId'],how='left')

# 1. product-client-depot
df_submit = df_pcd
ipcd = float(df_submit[df_submit.AdjDemand.notnull()].shape[0])/df_submit.shape[0]*100
pcd_miss = df_submit[df_submit.AdjDemand.isnull()]

# 2. Product-route
df_submit = df_submit.fillna(df_pr)
ipr = float(df_submit[df_submit.AdjDemand.notnull()].shape[0])/df_submit.shape[0]*100
pr_miss = df_submit[df_submit.AdjDemand.isnull()]

# 3. Client
df_submit = df_submit.fillna(df_c)
ic = float(df_submit[df_submit.AdjDemand.notnull()].shape[0])/df_submit.shape[0]*100
c_miss = df_submit[df_submit.AdjDemand.isnull()]

# # 4. Product-depot
df_submit = df_submit.fillna(df_pd)
ipd = float(df_submit[df_submit.AdjDemand.notnull()].shape[0])/df_submit.shape[0]*100
pd_miss = df_submit[df_submit.AdjDemand.isnull()]

# # 5. Product
df_submit = df_submit.fillna(df_p)
ip = float(df_submit[df_submit.AdjDemand.notnull()].shape[0])/df_submit.shape[0]*100
p_miss = df_submit[df_submit.AdjDemand.isnull()]

# # 6. Missing products
df_submit = df_submit.fillna(df_pmiss)

print rmsle(trueAdjDemand.values,df_submit.AdjDemand)

# ------------------------------------
print "Missing entries: {:4d}".format(df_submit[df_submit.AdjDemand.isnull()].shape[0])

print "pcd: {:6.2f}, pr: {:6.2f}, c: {:6.2f}, pd: {:6.2f}, p: {:6.2f}".\
		format(ipcd,ipr,ic,ipd,ip)