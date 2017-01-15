import pandas as pd
import numpy as np

# Training and Testing
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

#change column names for convenience
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']

# ------------------------------------
# Full match dataframe - will add to it and delete missing rows
cols = ['ProductId','ClientId','RouteId','DepotId','ChannelId']

df_train = df_train.set_index(cols)
df_test = df_test.set_index(cols)

#Sorting indices to speed-up
df_train.sortlevel(0,inplace=True)
df_train.sortlevel(1,inplace=True)
df_train.sortlevel(2,inplace=True)
df_train.sortlevel(3,inplace=True)
df_train.sortlevel(4,inplace=True)


df_result = pd.DataFrame(index=cols,\
		columns=['WeekNum','SalesUnitsWeek','SalesPesosWeek',\
				'ReturnsUnitsWeek','ReturnsPesosWeek','AdjDemand'])
# ------------------------------------
dfs = df_test.iloc[0:100]
match = 0
for i,row in dfs.iterrows():
	try:
		val = df_train.ix[i]
		match += 1
		df_result = pd.concat([df_result,val])
	except:
		pass


# ------------------------------------
# #Output
df_result = df_result.set_index('ProductId')
df_result.to_csv('processed/Product_Client_Route_Depot_Channel_match.csv')
# ------------------------------------

print "match: {:10.2f}".\
		format(float(match)/df_test.shape[0]*100)