import pandas as pd
import numpy as np
import re as re

df_town = pd.read_csv('input/town_state.csv')
df_town.columns = ['DepotId','Town','State']

df_town['depot_town_short'] = df_town.apply(lambda x: re.sub('(\d+\s)|(AG\.\s)|(AG\.)','',x.Town),axis=1)
df_town['depot_town_short'] = df_town.apply(lambda x: re.sub('^\s','',x.depot_town_short),axis=1)

agpt = df_town.groupby('depot_town_short').size().rename('depot_per_town').reset_index()
agps = df_town.groupby('State').size().rename('depot_per_state').reset_index()

df_town = pd.merge(df_town,agpt,on='depot_town_short',how='left')
df_town = pd.merge(df_town,agps,on='State',how='left')
df_town['depot_town'], fct = pd.factorize(df_town.depot_town_short)
df_town['depot_state'],fct = pd.factorize(df_town.State)
df_town = df_town[['DepotId','depot_town','depot_state','depot_per_town','depot_per_state']]


## Get mean demand values by depot from df_full
df_full = pd.read_csv("input/train.csv")
df_full.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                  'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                  'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
by_mean = df_full.groupby('DepotId').agg({'AdjDemand':np.mean}).reset_index()
by_mean = by_mean.rename(columns = {'AdjDemand':'depot_mean_demand'})
df_town = pd.merge(df_town,by_mean,on=['DepotId'],how='left')


store = pd.HDFStore('processed/processed_data.h5',complevel=9, complib='bzip2')
store['depots'] = df_town
store.close()