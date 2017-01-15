import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#scale data
dftrain = train_csv

#feature engineering
dftrain['demand_log'] = np.log(dftrain.Demanda_uni_equil+1)
(dftrain['Agencia_ID_fct'],Agencia_ID_fct) = pd.factorize(dftrain.Agencia_ID)
(dftrain['Cliente_ID_fct'],Cliente_ID_fct) = pd.factorize(dftrain.Cliente_ID)
(dftrain['Producto_ID_fct'],Producto_ID_fct) = pd.factorize(dftrain.Producto_ID)

#X and Y
X = dftrain[["Agencia_ID","Cliente_ID","Producto_ID"]]
Y = dftrain.demand_log


#random forest
forest = RandomForestRegressor(n_estimators=60)