import pandas as pd
import numpy as np

#read data
df_train = pd.read_csv("input/train.csv")
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

df_train['logAdjDemand'] = np.log(df_train.AdjDemand+1.)

# Get mean by product_client by week from df_train
by_product_client_week_mean = df_train.groupby(['ProductId','ClientId','WeekNum']).\
                            agg({'logAdjDemand':np.mean})

df = by_product_client_week_mean

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# Rolling mean
result = []
outliers = 0
for pid, _ in df.groupby(level = 0):
    for cid, this in _.groupby(level = 1):
        Yraw = np.array(this.values)

        Y = Yraw
        # Y = reject_outliers(Yraw)
        # if Y.shape[0] < Yraw.shape[0]:
            # outliers +=1

        # moving average
        N = Y.shape[0]
        if N > 2:
            val10ma = moving_average(Y)[-1:][0]
            val11ma = (val10ma+np.sum(Y[-2:]))/3
        elif N == 2:
            val10ma = np.sum(Y)/2.0
            val11ma = val10ma
        else:
            val10ma = Y[0][0]
            val11ma = val10ma
        
        # linear
        if N > 1:
            X = this.index.get_level_values('WeekNum').values
            Y = this.values
            Mx = np.mean(X)
            My = np.mean(Y)
            b = (np.sum(X*Y.reshape(1,-1))-N*Mx*My)/(np.sum(X**2)-N*Mx*Mx)
            a = My - b*Mx
            val10lin = (a+b*10)
            val11lin = (a+b*11)
            if val10lin < 0:
                # Fallback to a rolling 2-step mean
                val10lin = val10ma
            if val11lin < 0:
                # Fallback to a rolling 2-step mean
                val11lin = val11ma
        else:
            val10lin = val10ma
            val11lin = val11ma

        # convert to real values
        val10ma = np.exp(val10ma)-1.
        val11ma = np.exp(val11ma)-1.
        val10lin = np.exp(val10lin)-1.
        val11lin = np.exp(val11lin)-1.

        result.append({"ProductId":pid,"ClientId":cid,"WeekNum":10,"AdjDemandMA":val10ma,"AdjDemandLR":val10lin})
        result.append({"ProductId":pid,"ClientId":cid,"WeekNum":11,"AdjDemandMA":val11ma,"AdjDemandLR":val11lin})


# Rolling mean
# result = []
# outliers = 0
# for pid, _ in df.groupby(level = 0):
#     for cid, this in _.groupby(level = 1):
#         Yraw = np.array(this.values)

#         Y = reject_outliers(Yraw)
#         if Y.shape[0] < Yraw.shape[0]:
#             outliers +=1

#         N = Y.shape[0]
#         if N > 2:
#             val10 = moving_average(Y)[-1:][0]
#             val11 = (val10+np.sum(Y[-2:]))/3
#         elif N == 2:
#             val10 = np.sum(Y)/2.0
#             val11 = (val10+np.sum(Y[-1:]))/2
#         else:
#             val10 = np.mean(Yraw)
#             val11 = val10
#         result.append({"ProductId":pid,"ClientId":cid,"WeekNum":10,"MeanAdjDemand":val10})
#         result.append({"ProductId":pid,"ClientId":cid,"WeekNum":11,"MeanAdjDemand":val11})


# Linear with fallback to the rolling mean
# result = []
# for pid, _ in df.groupby(level = 0):
#     for cid, this in _.groupby(level = 1):
#         if this.shape[0] > 1:
#             N = this.shape[0]
#             X = this.index.get_level_values('WeekNum').values
#             Y = this.values
#             Mx = np.mean(X)
#             My = np.mean(Y)
#             b = (np.sum(X*Y.reshape(1,-1))-N*Mx*My)/(np.sum(X**2)-N*Mx*Mx)
#             a = My - b*Mx
#             val10 = (a+b*10)
#             val11 = (a+b*11)
#             if val10 < 0:
#                 # Fallback to a rolling 2-step mean
#                 val10 = np.mean(Y[-2:])
#             if val11 < 0:
#                 # Fallback to a rolling 2-step mean
#                 val11 = (Y[-1:][0]+val10)/2
#         else:
#             val10 = this.values[0][0]
#             val11 = val10
#         result.append({"ProductId":pid,"ClientId":cid,"WeekNum":10,"MeanAdjDemand":val10})
#         result.append({"ProductId":pid,"ClientId":cid,"WeekNum":11,"MeanAdjDemand":val11})

df_result = pd.DataFrame(result)
df_result.set_index('ProductId').to_csv('processed/product_client_by_week_log_mean.csv')

print "outliers: {:10.0f}".format(float(outliers))
print "outliers: %{:8.2f}".format(float(outliers)/df_result.shape[0]*100)


# # get median time prediction
# def predict_median_by_week(ProductId,ClientId,week):
#     key = tuple(row)
#     this = by_product_client_week.ix[row]
#     if this.shape[0] > 1:
#         N = this.shape[0]
#         Mx = np.mean(this.index.values)
#         My = np.mean(this.values)
#         b = (np.sum(this.index.values*this.values.reshape(1,-1))-N*Mx*My)/\
#             (np.sum(this.index.values**2)-N*Mx*Mx)
#         a = My - b*Mx
#         val = (a+b*week)
#         if val < 0:
#             val = 0
#     else:
#         val = this.values[0][0]
#     return val

# df_pcw = by_product_client_week

# df_pcw['PredDemandWeek10'] = df_pcw.apply(lambda x:\
#                             predict_median_by_week(x,10),axis=1)
# df_pcw['PredDemandWeek11'] = df_pcw.apply(lambda x:\
#                             predict_median_by_week(x,11),axis=1)

# df_pcw.set_index('ProductId').to_csv('input/processed_product_client_by_week.csv')