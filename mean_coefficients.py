import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Training and Testing
df = pd.read_csv("input/train.csv")
df.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

df_client = pd.read_csv("processed/processed_clients_cluster.csv")
df_client.columns = ['ClientId','ClientName','stems','ClusterId']

df_client = df_client[['ClientId','ClusterId']]
# df = pd.merge(df,df_client,on=['ClientId'],how='left')

# ---------------------------------------------------------
result = []
params = [0.8,0.5]

df_train = df[df.WeekNum >= 9]
df_test = df[df.WeekNum < 9]

df_train.AdjDemand = np.log1p(df_train.AdjDemand)

# choose the grouping 
keys = ['ProductId','RouteId']

# process and merge
by_group = df_train.groupby(keys).agg({'AdjDemand': np.mean}).reset_index()
by_group.AdjDemand = np.expm1(by_group.AdjDemand)
df_test = pd.merge(df_test,by_group,on=keys,how='left')

df_test = df_test[df_test.AdjDemand_y.notnull()]

# objective function for minimization
def obj_function(params):
	pred = df_test.AdjDemand_y*params[0]+params[1]
	true = df_test.AdjDemand_x
	return rmsle(true,pred)

def rmsle(true,predict):
    logp = np.log1p(predict)
    logt = np.log1p(true)
    delta = np.sqrt(np.sum((logp-logt)**2)/logp.shape[0])
    return delta


res = minimize(obj_function, params, method='Nelder-Mead',tol=1.0e-3,
                    options={'disp': True, 'maxiter': 120})

print res

# for i in range(3,10):
# 	df_train = df[df.WeekNum != i]
# 	df_test = df[df.WeekNum == i]

# 	df_train.AdjDemand = np.log1p(df_train.AdjDemand)

# 	# choose the grouping 
# 	keys = ['Product','ClientId','DepotId']

# 	# process and merge
# 	by_group = df_train.groupby(keys).agg({'AdjDemand': np.mean}).reset_index()
# 	by_group.AdjDemand = np.expm1(by_group.AdjDemand)
# 	df_test = pd.merge(df_test,by_group,on=keys,how='left')
	
# 	df_test = df_test[df_test.AdjDemand_y.notnull()]

# 	# objective function for minimization
# 	def obj_function(params):
# 		pred = df_test.AdjDemand_y*params[0]+params[1]
# 		true = df_test.AdjDemand_x
# 		return rmsle(true,pred)

# 	def rmsle(true,predict):
# 	    logp = np.log1p(predict)
# 	    logt = np.log1p(true)
# 	    delta = np.sqrt(np.sum((logp-logt)**2)/logp.shape[0])
# 	    return delta


# 	res = minimize(obj_function, params, method='Nelder-Mead',tol=1.0e-3,
# 	                    options={'disp': True, 'maxiter': 120})
	
# 	result.append({'week':i,'params':res.x,'fun':res.fun})

# print result