import pandas as pd
import numpy as np

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
producto_tabla_csv = pd.read_csv("producto_tabla.csv")
town_state_csv = pd.read_csv("town_state.csv")
cliente_tabla_csv = pd.read_csv("cliente_tabla.csv")
sample_submission_csv = pd.read_csv("sample_submission.csv")

df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

df_test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId']

#TODO: XGBOOST
