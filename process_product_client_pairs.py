import pandas as pd
import numpy as np

# Training and Testing
df_test = pd.read_csv("input/test.csv")
df_train = pd.read_csv("input/train.csv")

#change column names for convenience
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']
df_test = df_test[['ProductId','ClientId','WeekNum']]

# ------------------------------------
df_pcl = pd.read_csv("processed/mean_product_client_by_week.csv")
df_pcl = df_pcl[['ProductId','ClientId','WeekNum']]
# ------------------------------------

intersection = pd.merge(df_test, df_pcl, on=['ProductId', 'ClientId', 'WeekNum'], how='inner')
intersection.set_index(['ProductId']).to_csv('processed/product_client_pairs.csv')

# df_pcl_keys = np.array([0,0,0])
# pcl = 0

# for index, row in df_test.iterrows():
# 	# key = tuple(row['ProductId','ClientId','WeekNum'])
# 	# first try to match client and product pair
# 	try:
# 		val = df_pcl.ix[index]
# 		pcl +=1
# 		df_pcl_keys = np.vstack((df_pcl_keys,index))
# 	# if fails - use only product
# 	except:
# 		pass

# # # ------------------------------------
# # Process product client pairs
# # a = np.asarray(df_pcl_keys)
# np.savetxt("processed/product_client_pairs.csv", df_pcl_keys, fmt="%d",delimiter=",")

# # ------------------------------------
# print "pcl: {:10.2f}".format(float(pcl)/df_test.shape[0]*100)

# print "pcl: {:10.0f}".format(float(pcl))