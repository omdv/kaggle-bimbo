import pandas as pd
import numpy as np

# Training and Testing
# df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

#change column names for convenience
# df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
#                     'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
#                     'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']

# df_train["PricePiece"] = df_train.SalesPesosWeek/df_train.SalesUnitsWeek

# ------------------------------------
# Load Processed or Predicted Products
# df_product = pd.read_csv("processed/predicted_products.csv")
# Get dictionaries based on processed products
# by_product = df_product[['ID','AdjDemand_median']].set_index('ID').to_dict()


# ------------------------------------
df_pcl = pd.read_csv("processed/product_client_by_week_moving_mean.csv")
by_product_client = df_pcl.set_index(['ProductId','ClientId','WeekNum']).to_dict()

# Get dictionary from df_train
# by_product_client = df_train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean}).to_dict()
# by_pcl_week = df_pcl.set_index(['ProductId','ClientId']).to_dict()

# ------------------------------------

df_pcl_keys = []
pcl = 0
pwk = 0

# Assign target values
def process_df(row):
	global pcl, pwk, glm, df_pcl_keys
	key = tuple(row)
	# first try to match client and product pair
	try:
		val = by_product_client['MeanAdjDemand'][key]
		pcl +=1
		df_pcl_keys.append(key)
	# if fails - use only product
	except:
		val = -1
		# val = by_product['AdjDemand_mean'][row[0]]
		# val = by_product['AdjDemand_median'][row[0]]
		# pwk +=1
	return val

# ------------------------------------
# Process the dataframe
df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId', 'WeekNum']].\
                apply(lambda row:process_df(row), axis=1)

# # ------------------------------------
# Process product client pairs
a = np.asarray(df_pcl_keys)
np.savetxt("processed/product_client_pairs.csv", a, fmt="%d",delimiter=",")

# # ------------------------------------
# # Save submission file
# # case_name = 'mean_pcl_by_week_median_predicted_product'

# # # #Output
# # df_submit = df_test[['id', 'Demanda_uni_equil']]
# # df_submit = df_submit.set_index('id')
# # df_submit.to_csv('output/'+case_name)

# ------------------------------------
print "pcl: {:10.2f}, pwk: {:10.2f}".\
		format(float(pcl)/df_test.shape[0]*100,float(pwk)/df_test.shape[0]*100)

print "pcl: {:10.0f}, pwk: {:10.0f}".\
		format(float(pcl),float(pwk))