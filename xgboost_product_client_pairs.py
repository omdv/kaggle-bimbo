import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split

# Training and Testing
df_train = pd.read_csv('processed/product_client.csv')
df_test = pd.read_csv('processed/product_client_pairs.csv')

#Define new Y
df_train['logAdjDemand'] = np.log(df_train.AdjDemand+1.0)

#Define X and Y
X = df_train[['ProductId','ClientId','WeekNum_y']]
y = df_train['logAdjDemand']


#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,\
									test_size=0.33, random_state=42)


#XGBoost
params = {"objective": "reg:linear",
          "eta": 0.1,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params,\
		xgb.DMatrix(X_train, y_train),\
		num_trees)

#Predict and get a score
y_pred = gbm.predict(xgb.DMatrix(X_test))
score = np.sqrt(np.sum((y_pred-y_test)**2)/y_pred.shape[0])