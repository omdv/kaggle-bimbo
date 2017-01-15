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

df_train["PricePiece"] = df_train.SalesPesosWeek/df_train.SalesUnitsWeek
# ------------------------------------


# Load Processed or Predicted Products
df_product = pd.read_csv("processed/predicted_products.csv")
df_pcl = pd.read_csv("processed/product_client_by_week_moving_mean.csv")

# ------------------------------------

#General Mean
general_median = df_train.AdjDemand.median()

# Get dictionaries based on processed products
# by_product = df_product[['ID','AdjDemand_median']].set_index('ID').to_dict()
by_product = df_product[['ID','AdjDemand_median']].set_index('ID').to_dict()


# Get dictionary from df_train
# by_product_client = df_train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean}).to_dict()
by_pcl_week = df_pcl.to_dict()

# ------------------------------------

pcl = 0
pwk = 0
glm = 0

matches = 0
missing = []

# Assign target values
def process_df(row):
	global matches, missing, pcl, pwk, glm
	key = tuple([row[0],row[1]])
	# first try to match client and product
	try:
		val = by_product_client['AdjDemand'][key]
		pcl +=1
	# if fails - use time_series for a corresponding week for product
	except:
		# val = by_product['AdjDemand_mean'][key[0]]
		val = by_product['AdjDemand_median'][key[0]]
		pwk +=1
	return val

# ------------------------------------
# Process the dataframe
df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId', 'WeekNum']].\
                apply(lambda row:process_df(row), axis=1)
# ------------------------------------


# #Output
df_submit = df_test[['id', 'Demanda_uni_equil']]
df_submit = df_submit.set_index('id')
df_submit.to_csv('output/mean_pcl_pred_product_global_median.csv')

# ------------------------------------

# Save Missing names
df_missing = pd.DataFrame(missing, columns = ['ProductId','ClientId']).set_index('ProductId')
df_missing.to_csv('output/mean_pcl_pred_product_global_median_missing.csv')

print "pcl: {:10.2f}, pwk: {:10.2f}, glm: {:10.2f}".\
		format(float(pcl)/df_submit.shape[0]*100,\
		float(pwk)/df_submit.shape[0]*100,\
		float(glm)/df_submit.shape[0]*100)

print "pcl: {:10.0f}, pwk: {:10.0f}, glm: {:10.0f}".\
		format(float(pcl),float(pwk),float(glm))