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


# Load Processed or Predicted Products
df_product = pd.read_csv("input/processed_products.csv")
df_product_client_week = pd.read_csv("input/processed_product_client_by_week.csv")

# ------------------------------------

#General Mean
general_mean = df_train.AdjDemand.mean()

# Get dictionaries based on processed products
by_product = df_product[['ID','PredDemandWeek10','PredDemandWeek11']].set_index('ID').to_dict()
by_product_client = df_product_client_week.set_index(['ProductId','ClientId']).to_dict()


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
	try:
		# first try to match client and product
		try:
			if row[2] == 10:
				val = by_product_client['PredMedianDemandWeek10'][key]
			if row[2] == 11:
				val = by_product_client['PredMedianDemandWeek11'][key]
			pcl +=1
		# if fails - use time_series for a corresponding week for product
		except:
			if row[2] == 10:
				val = by_product['PredDemandWeek10'][key[0]]
			if row[2] == 11:
				val = by_product['PredDemandWeek11'][key[0]]
			pwk +=1
	except:
		# if all fails - general mean
		glm +=1
		missing.append(key)
		val = general_mean
	return val

# ------------------------------------
# Process the dataframe
df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId', 'WeekNum']].\
                apply(lambda row:process_df(row), axis=1)
# ------------------------------------


# #Output
df_submit = df_test[['id', 'Demanda_uni_equil']]
df_submit = df_submit.set_index('id')
df_submit.to_csv('output/product_client_median_by_week.csv')

# ------------------------------------

# Save Missing names
df_missing = pd.DataFrame(missing, columns = ['ProductId','ClientId']).set_index('ProductId')
df_missing.to_csv('output/product_client_median_02_missing.csv')

print "pcl: {:10.2f}, pwk: {:10.2f}, glm: {:10.2f}".\
		format(float(pcl)/df_submit.shape[0]*100,\
		float(pwk)/df_submit.shape[0]*100,\
		float(glm)/df_submit.shape[0]*100)

print "pcl: {:10.0f}, pwk: {:10.0f}, glm: {:10.0f}".\
		format(float(pcl),float(pwk),float(glm))