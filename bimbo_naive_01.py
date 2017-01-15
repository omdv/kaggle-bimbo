import pandas as pd
import numpy as np

#read data
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")
# df_product = pd.read_csv("processed/predicted_products.csv")
df_pcl = pd.read_csv("processed/processed_product_client_by_week.csv")

#change column names for convenience
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']
# df_product.columns = ['ProductId','productTypeFct','brandFct','weight_s','volume_s','pieces_s',\
# 					'wpp_s','has_choco','has_vanilla','has_multigrain','MedianAdjDemand',\
# 					'medianDemand_s']


#Based on mean values
general_median = df_train.AdjDemand.median()
by_product = df_train.groupby('ProductId').agg({'AdjDemand':np.mean}).to_dict()
# by_product_client = df_train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean}).to_dict()
by_product = df_train.groupby('ProductId').agg({'AdjDemand':np.median}).to_dict()
by_product_client = df_train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median}).to_dict()

#new by_product from predictions
# by_product = df_product[['ProductId','MedianAdjDemand']].to_dict()

pcl = 0
pwk = 0
glm = 0

missing = []

# Assign target values
def process_df(row):
	global missing, pcl, pwk, glm
	key = tuple([row[0],row[1]])
	try:
		# first try to match client and product
		try:
			val = by_product_client['AdjDemand'][key]
			pcl +=1
		# if fails - use time_series for a corresponding week for product
		except:
			val = by_product['AdjDemand'][key[0]]
			pwk +=1
	except:
		# if all fails - general mean
		glm +=1
		missing.append(key)
		val = general_median
	return val

# #Process the dataframe
df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId']].\
                apply(lambda row:process_df(row), axis=1)

# ------------------------------------

# Save Missing names
df_missing = pd.DataFrame(missing, columns = ['ProductId','ClientId']).set_index('ProductId')
df_missing.to_csv('output/naive_median_missing_products.csv')

print "pcl: {:10.2f}, pwk: {:10.2f}, glm: {:10.2f}".\
		format(float(pcl)/df_test.shape[0]*100,\
		float(pwk)/df_test.shape[0]*100,\
		float(glm)/df_test.shape[0]*100)

print "pcl: {:10.0f}, pwk: {:10.0f}, glm: {:10.0f}".\
		format(float(pcl),float(pwk),float(glm))