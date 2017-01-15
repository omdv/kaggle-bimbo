import pandas as pd
import numpy as np
import re

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# function to tokenize product names
def words_to_stems(sen,remove_modifiers=False):
    stemmer = SnowballStemmer("spanish")
    filtered_words = []
    stems = []
    
    sen = sen.lower()
    sen = re.sub(u'\(.*\)','',sen)
#    sen = re.sub(u'(\d+(kg|ml|pct|pq|g|p))','',sen)
    if remove_modifiers:    
        sen = re.sub(u'super|hot|mini|div','',sen)
        sen = re.sub(u'multigrano|integral|prom|blanco|especial|clasico|grande|chocolate','',sen)
        sen = re.sub(u'intenso','',sen)
        sen = re.sub(u'vainilla|clasicas|int|whole|grain|multigr|chocol|choco|edicion','',sen)
        sen = re.sub(u'mediano|tubo|doble|fibra|clasica|resellable|natural|karamelo|extra','',sen)
        sen = re.sub(u'original|chochitos','',sen)
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sen)
    
    # filter out non-alpha words
    for word in words:
       if re.search('^[a-zA-Z]', word):
           filtered_words.append(word)
    
    # stemming
    for w in filtered_words:
        stems.append(stemmer.stem(w))
           
    return ' '.join(stems)
# ---------------------------------------------------------
# # read train

# df_full = pd.read_csv("input/train.csv")
 df_full.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

# read pickled product data
#df_product = pd.read_hdf('processed/processed_data.h5','products')

 # # Or continue from previous point - ran by R
# df_product = pd.read_csv("processed/processed_products.csv")
# del df_product['Unnamed: 0']
# df_product = df_product[df_product.ProductId != 0]
# ---------------------------------------------------------

## Get mean demand values by product from df_full
by_mean = df_full.groupby('ProductId').agg({'AdjDemand':np.mean}).reset_index()
by_mean = by_mean.rename(columns = {'AdjDemand':'product_mean_demand'})
df_product = pd.merge(df_product,by_mean,on=['ProductId'],how='left')

# # Get product price
# df_full['product_price'] = df_full.SalesPesosWeek/df_full.SalesUnitsWeek
# by_price = df_full.groupby('ProductId').agg({'product_price':np.mean}).reset_index()
# df_product = pd.merge(df_product,by_price,on=['ProductId'],how='left')

# Process short name
df_product['product_stems'] =\
    df_product.apply(lambda row: words_to_stems(row.product_shortname),axis=1)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_product.product_stems)

 cls = KMeans(n_clusters = 2591)
 km = cls.fit_predict(X)
 df_product['product_shortname_tfidf_2591'] = km

#cls = KMeans(n_clusters = 512)
#km = cls.fit_predict(X)
#df_product['product_name_tfid_512'] = km.astype(np.int32)
#
#cls = KMeans(n_clusters = 60)
#km = cls.fit_predict(X)
#df_product['product_name_tfid_60'] = km.astype(np.int32)

# cls = KMeans(n_clusters = 30)
# km = cls.fit_predict(X)
# df_product['product_name_cluster_30'] = km

# # Process product type, which is name without modifiers
# # df_product['product_type'] =\
# #     df_product.apply(lambda row:\
# #     words_to_stems(row.product_shortname,True),axis=1)

 # Product perk - Dima's version
 perk_brand = df_product['product_name'].str.extract('\s((\D|CR\d)+)\s\d+$', expand=False)[0].fillna('IDENTIFICADO').str.split(' ')
 df_product['product_perks'] = perk_brand.apply(lambda x: ''.join(x[-2:-1]))

# Fill product volume
#df_product.product_weight = df_product['product_weight'].fillna(df_product.product_volume)

# ---------------------------------------------------------
# K-mean grid search
# km_x = []
# km_y = []
# for i in [5,10,20,30,40,50,60,70,80,90,100,110]:
#     cls = KMeans(n_clusters=i)
#     cls.fit(X)
#     km_x.append(i)
#     km_y.append(cls.score(X))
# ---------------------------------------------------------


#store = pd.HDFStore('processed/processed_data.h5',complevel=9, complib='bzip2')
#store['products'] = df_product
#store.close()