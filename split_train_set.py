import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# Training and Testing
df_train = pd.read_csv("input/train.csv")


#change column names for convenience
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

by_pcl = df_train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean})
by_pcl = by_pcl.reset_index()[['ProductId','ClientId']]

X_train, X_test = train_test_split(by_pcl, test_size=0.33, random_state=42)

pcl_train = pd.merge(X_train,df_train,on=['ProductId','ClientId'],how='inner')
pcl_test = pd.merge(X_test,df_train,on=['ProductId','ClientId'],how='inner')

print float(pcl_train.shape[0]+pcl_test.shape[0])/df_train.shape[0]

# #Output
# df_submit = df_test[['id', 'Demanda_uni_equil']]
# df_submit = df_submit.set_index('id')
# df_submit.to_csv('output/product_client_median_by_week.csv')

# ------------------------------------