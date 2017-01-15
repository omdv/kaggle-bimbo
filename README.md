# Default Median Naive
The breakdown of cases is shown below. In 80% of the cases there a product/client pair, 19% of cases - only product exists in the training set and only in the case 0.37% there was no product.
<pre>
pcl:      80.49, pwk:      19.14, glm:       0.37
pcl:    5634038, pwk:    1339573, glm:      25640
</pre>

96.5% cases there were zero returns
<pre>
In [26]: float(df_train[df_train.ReturnsUnitsWeek==0].shape[0])/df_train.shape[0]
Out[26]: 0.9656990417315265
</pre>

97% cases there was a zero delta between Sales This Week and Adjusted Demand
<pre>
In [27]: float(delta[delta.values==0].count())/delta.shape[0]
Out[27]: 0.9705457511173405
</pre>

# Analysis:
<pre>
In [49]: pcrdc = df_train.groupby(['ProductId','ClientId','RouteId','DepotId','ChannelId','WeekNum']).agg({'AdjDemand':np.mean})
In [51]: pcrdc.shape[0]-df_train.shape[0]
Out[51]: 0
</pre>

Channel impact: negligible
<pre>
In [52]: pcrdw = df_train.groupby(['ProductId','ClientId','RouteId','DepotId','WeekNum']).agg({'AdjDemand':np.mean})
In [53]: pcrdw.shape[0]-df_train.shape[0]
Out[53]: -906
</pre>

Depot impact: 0.13% insignificant
<pre>
In [57]: pcrcw = df_train.groupby(['ProductId','ClientId','RouteId','ChannelId','WeekNum']).agg({'AdjDemand':np.mean})
In [60]: float(pcrcw.shape[0]-df_train.shape[0])/df_train.shape[0]*100
Out[60]: -0.13263195549706996
</pre>

Route impact: 0.17% insignificant
<pre>
In [61]: pcdcw = df_train.groupby(['ProductId','ClientId','DepotId','ChannelId','WeekNum']).agg({'AdjDemand':np.mean})
In [63]: float(pcdcw.shape[0]-df_train.shape[0])/df_train.shape[0]*100
Out[63]: -0.17520920332879017
</pre>

Client impact: 92% as discussed
<pre>
In [64]: prdcw = df_train.groupby(['ProductId','RouteId','DepotId','ChannelId','WeekNum']).agg({'AdjDemand':np.mean})
In [65]: float(prdcw.shape[0]-df_train.shape[0])/df_train.shape[0]*100
Out[65]: -92.77004252763909
</pre>

PCD coefficients with one week as cross-validation:
<pre>
{'week': 3, 'fun': 0.4554159258436643, 'params': array([ 0.82600269,  0.64759766])}, 
{'week': 4, 'fun': 0.45404479107764695, 'params': array([ 0.85819428,  0.51106624])}, 
{'week': 5, 'fun': 0.45386780264614035, 'params': array([ 0.87099413,  0.46620465])}, 
{'week': 6, 'fun': 0.45827593864536498, 'params': array([ 0.84775699,  0.46533671])}, 
{'week': 7, 'fun': 0.46296447744446129, 'params': array([ 0.86795431,  0.42561801])}, 
{'week': 8, 'fun': 0.46616527066198538, 'params': array([ 0.83354196,  0.46582455])}, 
{'week': 9, 'fun': 0.46952075559983836, 'params': array([ 0.82147681,  0.45916909])}
all: array([ 0.10351427,  0.93634687]), fun: 0.20679
</pre>

Client coefficients with one week as cross-validation:
<pre>
{'week': 3, 'fun': 0.69866595489186267, 'params': array([ 0.92928417,  0.24374962])}, 
{'week': 4, 'fun': 0.70973194839066089, 'params': array([ 0.96979459,  0.16377323])}, 
{'week': 5, 'fun': 0.71932850990413411, 'params': array([ 0.95979195,  0.20729744])}, 
{'week': 6, 'fun': 0.7251538478743057, 'params': array([ 0.91676292,  0.24654381])}, 
{'week': 7, 'fun': 0.73548834171475375, 'params': array([ 0.93070802,  0.2730925 ])}, 
{'week': 8, 'fun': 0.73200887483562893, 'params': array([ 0.91307282,  0.29342228])}, 
{'week': 9, 'fun': 0.72750636777272393, 'params': array([ 0.90754417,  0.271403  ])}
all: array([ 0.09730284,  1.05209192]), fun: 0.2822
</pre>

Product-Cluster coefficients with week 9 as cross-validation
<pre>
[ 0.86921662,  0.57833923], fun: 0.64907
</pre>

# Filtering
Filtering product/client pairs by week with 4 sigma normal - 24.25% of pairs have outliers.

# History of submissions by model
* 0.76500 - naive mean by product ID
* 0.50774 - naive, median by client and product, then by product, then global mean
* 0.50778 - naive, median by client and product, then by product, then random forest predictions
* 0.50600 - median by product/client, time-series by product, then global mean
* 0.67060 - median by product/client and product, time series, wrong try/except. No improvement on time series.
* 0.50149 - mean by product/client (81%), median predicted Random Forest product (19%)
* 0.54526 - mean by product client (81%), mean predicted Random Forest product (19%).
* 0.67076 - product/client - mean linear regression by week (80.49%), product - median, predicted by forest (19.51%). See the similar example above - almost same score.
* 0.50815 - rolling mean by product/client(81%), median predicted Random Forest product (19%)
* 0.50746 - same as above but with filtering on 4 sigma normal
* 0.50140 - filtered product/client mean by week, median predicted random forest
* 0.50491 - rolling mean of log(pcl+1)-0.91, median predicted RF
* 0.50399 - rolling mean of log(pcl+1)-1, median RF for product
* 0.48873 - pcd: 80.48, pc: 80.49, pr: 97.69, pd: 96.28, p: 100.00, all log mean, except median product
* 0.49034 - pcdr:  79.34, pcd:  80.48, pc:  80.49, pr:  97.70, pd:  97.85, pch:  99.63, p: 100.00
* 0.47669 - pcd: log-mean 0.83 + 0.44, pr: log-mean 0.79+0.09, product - predicted mean
* 0.47615 - pcd: log-mean 0.826 + 0.45, pr: log-mean: 0.79+0.09, product: 1.05+0.6, missing - predicted product
* 0.47495 - added mean by client pcd: 80.48, pr:  97.70, c:  99.99, pd:  99.99, p: 100.00
* 0.49297 - added mean by product-cluster combination pcd:  80.48, pcc:  98.43, pr:  98.57, c: 100.00, pd: 100.00, p: 100.00
* 0.48776 - pcd, then pr, then pcc. pcd:  80.48, pr:  97.70, pcc:  98.57, c: 100.00, pd: 100.00, p: 100.00

# XGBoost history
* naive on week 8 as is with predictions on week 9 -> 0.7106
* training in log1p() - 0.6421
* added product-client 4 week history, 0.4879 on week 9
<pre>
ProductId    16.124031
DepotId      14.726154
RouteId      14.288844
Hist_4       10.821537
Hist_1       10.782777
Hist_3        9.860128
Hist_2        9.598921
ClientId      9.344456
ChannelId     4.453151
</pre>
* added product features. 0.493 with 50 estimators, max_depth = 8, lr = 0.05
<pre>
Hist_2                      15.350212
Hist_4                      14.375849
Hist_3                      14.040412
Hist_1                      11.173229
Product_weight               8.337992
Product_AdjDemandMean        5.630541
Product_type_fct             4.951681
ChannelId                    4.839869
RouteId                      4.520406
DepotId                      3.138727
ProductId                    3.074834
Product_weight_per_piece     2.475841
Product_cluster              1.765035
Product_brand_fct            1.749062
ClientId                     1.605303
Product_pieces               1.573357
Product_has_promotion        0.910470
Product_volume               0.191678
Product_has_choco            0.159732
Product_has_vanilla          0.095839
Product_has_multigrain       0.039933
</pre>
* same as above with lr = 0.1 and 128 estimators, rmsle = 0.4715
* same as above with Depot_by_town, Depot_by_state and client cluster = 0.4702, 256 estimators
<pre>
Hist_1                      9.046773
DepotId                     8.431256
Hist_4                      7.942464
Hist_3                      7.662684
RouteId                     7.368092
Hist_2                      6.643955
Product_weight              6.420131
Product_AdjDemandMean       6.324677
Depot_per_state             5.241763
ProductId                   5.065666
Product_type_fct            4.603206
ChannelId                   3.806655
ClientId                    3.544979
Product_cluster             3.143412
Depot_per_town              3.005168
Product_weight_per_piece    2.927817
Product_brand_fct           2.498272
Client_cluster              2.452191
Product_pieces              2.035812
Product_has_promotion       0.990751
Product_has_choco           0.286363
Product_volume              0.210658
Product_has_vanilla         0.182680
Product_has_multigrain      0.164577
</pre>

<pre>
Predicting Week 9:
(10408713, 34)

Model Report:
RMSE (train): 0.4429
RMSE (test): 0.4488

Features (cross-validation):
hist_pcdrc_1                7.795998
DepotId                     7.040277
hist_pcdrc_2                6.596087
hist_pcdrc_3                6.558429
hist_pcdrc_4                6.026942
client_mean_demand          5.707708
RouteId                     5.693158
depot_mean_demand           5.023023
product_mean_demand         4.940860
depot_per_state             4.688383
product_weight              4.142346
ProductId                   3.643382
ClientId                    3.363517
product_price               3.130724
product_wpp                 2.740453
product_name_cluster_max    2.629192
client_cluster_1024         2.239777
product_name_cluster_120    2.226939
depot_per_town              2.214101
product_pieces              2.198696
product_name_cluster_60     2.195272
ChannelId                   1.791308
product_name_cluster_30     1.649236
client_cluster_128          1.507164
product_perks_fct           1.468650
product_brand_fct           1.413019
product_has_promotion       0.469009
product_has_choco           0.225946
product_volume              0.222523
product_has_integral        0.195135
product_has_multigrain      0.143784
product_has_vanilla         0.118964
dtype: float64
</pre>

* New history features
hist_route, hist_depot, hist_channel are removed
<pre>
Model Report:
RMSE (train): 0.454
RMSE (test): 0.4581

Features (cross-validation):
client_mean_demand          10.611698
hist_pcdrc_1                 5.680216
hist_pcdrc_2                 5.212642
hist_pcdrc_3                 5.029310
hist_pcdrc_4                 4.849829
product_weight               4.798989
hist_prodclient_4            4.524761
hist_prodclient_1            4.441569
hist_prod_1                  4.121893
hist_prodclient_2            3.927776
hist_client_1                3.889261
hist_prodclient_3            3.844584
hist_client_4                3.377780
hist_client_3                2.982614
hist_prod_4                  2.808526
hist_prod_2                  2.754604
hist_client_2                2.712238
RouteId                      2.682966
product_price                2.453416
product_name_cluster_max     2.203067
hist_channel_1               1.819456
hist_prod_3                  1.793266
ProductId                    1.721628
depot_state                  1.569878
product_pieces               1.248662
DepotId                      0.930526
hist_channel_2               0.794183
ChannelId                    0.790331
depot_town                   0.713301
depot_per_state              0.567714
product_mean_demand          0.475277
hist_route_1                 0.445235
hist_route_4                 0.444465
hist_channel_3               0.440613
hist_depot_1                 0.382070
hist_route_3                 0.345096
hist_depot_2                 0.338163
hist_route_2                 0.332001
hist_depot_4                 0.330460
ClientId                     0.323527
client_cluster_1024          0.297337
hist_depot_3                 0.295796
depot_mean_demand            0.282701
depot_per_town               0.194116
product_wpp                  0.134803
hist_channel_4               0.081652
</pre>


* with tfid clusters
<pre>
Training on Full set:
(31198430, 39)

Model Report:
RMSE (train): 0.4372

Features (full set):
client_mean_demand       4.981201
hist_pcdrc_1             4.620561
product_weight           4.309135
RouteId                  3.854905
hist_client_1            3.570910
hist_client_4            3.451503
hist_prodclient_1        3.300631
hist_client_3            3.258677
depot_mean_demand        3.234473
depot_state              3.208655
hist_client_2            3.165895
hist_prodruta_1          3.158634
DepotId                  3.139270
hist_pcdrc_3             3.134430
hist_pcdrc_2             3.119100
hist_prod_1              2.961774
product_price            2.931115
ProductId                2.919013
hist_pcdrc_4             2.885127
hist_prodclient_4        2.593872
depot_town               2.477692
product_name_tfid_512    2.418795
hist_prod_4              2.413148
hist_prodclient_3        2.309877
hist_prod_2              2.308263
hist_prodclient_2        2.280832
hist_prodruta_2          2.195311
hist_prodruta_4          2.107369
hist_prod_3              2.062188
product_pieces           1.895987
hist_prodruta_3          1.854033
depot_per_state          1.833056
ClientId                 1.821761
ChannelId                1.327191
client_tfid_argmax       1.206977
depot_per_town           1.040776
product_mean_demand      0.647863
</pre>


# Feature ideas
- history by other pairs
- number of routes per depot
- number of routes per client
- number of clients per product
- number of products per client
- total sales per client

# Product analysis
New products to go to the market
[37494, 37495, 36673, 37202, 37688, 37404, 46131, 37626, 37620, 37362, 36524, 32421, 37496, 32820, 98, 31203, 37617, 48217, 46064, 33053, 37702, 37405, 37610, 31655, 32798, 37618, 31211, 37745, 32224, 32591, 42323, 35191, 35246, 32026]
Only 0.36% of the test dataset

# Issues
id 3127280 is missing?

# References
http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf
https://www.quora.com/Data-Science-Can-machine-learning-be-used-for-time-series-analysis
https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.8923

Matching on week 10:
<pre>
df_week11[df_week11.hist_pcdrc_1.isnull()].shape[0]/0.01/df_week11.shape[0]
77.9884572242901 - on week 10

df_week11[df_week11.hist_pcdrc_2.isnull()].shape[0]/0.01/df_week11.shape[0]
60.50601207905767
50.13057425511418 - on week 9

df_week11[df_week11.hist_pcdrc_3.isnull()].shape[0]/0.01/df_week11.shape[0]
67.71443910281415
60.50601207905767 - on week 9

df_week11[df_week11.hist_pcdrc_4.isnull()].shape[0]/0.01/df_week11.shape[0]
70.09921794140541
67.71443910281415 - on week 9
</pre>