import pandas as pd
import numpy as np

# Training and Testing
df_test = pd.read_csv("input/test.csv")

#change column names for convenience
df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']

# ------------------------------------
# Load Processed or Predicted Products/Clients
df_product = pd.read_csv("processed/predicted_products.csv")
df_pcl = pd.read_csv("processed/product_client_by_week_moving_mean.csv")

# ------------------------------------
df_test = df_test[['id','WeekNum','ProductId','ClientId']]
df_pcl.columns = ['ProductId','ClientId','AdjDemand','WeekNum']

df_product = df_product[['ID','AdjDemand_median']]
df_product.columns = ['ProductId','AdjDemand']

# ------------------------------------

# Get ProductClient DataFrame
pcl = pd.merge(df_pcl,df_test,on=['ProductId','ClientId','WeekNum'],how='inner')


# Get Product Dataframe
idx_test = df_test.set_index('id').index
idx_pcl = pcl.set_index('id').index
idx_prd = idx_test.difference(idx_pcl)

df_test_product = df_test.ix[idx_prd]
prd = pd.merge(df_test_product,df_product,on=['ProductId'],how='inner')



# ------------------------------------
# Save submission file
case_name = 'filtered_mean_pcl_pred_product.csv'

# Output
df_submit = pd.concat([prd,pcl])
df_submit = df_submit[['id','AdjDemand']].sort_values(by='id')
df_submit.columns = ['id', 'Demanda_uni_equil']
df_submit = df_submit.set_index('id')
df_submit.to_csv('output/'+case_name)

# ------------------------------------
print "pcl: {:10.2f}, pwk: {:10.2f}".\
		format(float(pcl.shape[0])/df_test.shape[0]*100,
				float(prd.shape[0])/df_test.shape[0]*100)

print "pcl: {:10.0f}, pwk: {:10.0f}".\
		format(float(pcl.shape[0]),float(prd.shape[0]))