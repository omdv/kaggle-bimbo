import pandas as pd
import numpy as np
import re

#read data
df_product = pd.read_csv("input/processed_products.csv")
df_train = pd.read_csv("input/train.csv")
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

#Get median values by product
by_product = df_train.groupby('ProductId').agg({'AdjDemand':np.median}).to_dict()

def find_median(row):
    try:
        val = by_product['AdjDemand'][row.ID]
    except:
        val = -1.0
    return val

df_product['MedianAdjDemand'] = df_product.apply(lambda row: find_median(row),axis=1)
df_product.to_csv('input/processed_products.csv')