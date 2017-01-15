import pandas as pd
import numpy as np
import re

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

def sparse_argmax(X):
    y = []
    for i in range(X.shape[0]):
        	y += [X[i].toarray().argmax()]
    return y

#read data
df_client = pd.read_csv("input/cliente_tabla.csv")
df_client.columns = ['ClientId','ClientName']
df_client = df_client.drop_duplicates('ClientId')

# function to tokenize product names
def word_tokenize_stem(row):
    value = row['ClientName']
    sen = unicode(value,'utf-8')
    
    # try:
    #     sen = unicode(value, "ascii")
    # except UnicodeError:
    #     sen = unicode(value, "utf-8")
    # else:
    #     pass
    
    stemmer = SnowballStemmer("spanish")
    filtered_words = []
    stems = []
    
    predefined = [\
    'UNIVERSIDAD',\
    'WAL MART',\
    'HOTEL',\
    'CAFE',\
    'PAPELERIA',\
    'PASTELERIA',\
    'PANADERIA',\
    'CASA',\
    'BODEGA',\
    'BIMBO',\
    'PASADITA',\
    'PERLITA',\
    'ESCUELA',\
    'CARNICERIA',\
    'PREPARATORIA',\
    'RESTAURANTE',\
    '7 ELEVEN',\
    'OXXO',\
    'HOSPITAL',\
    'SIN NOMBRE',\
    'INSTITUTO'\
    ]
    
    for word in predefined:
        if re.search(word,sen):
            sen = word
            sen = re.sub(u'\s','',sen)
    
    # remove FMA\d+    
    sen = re.sub(u'FMA\d+','',sen)
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sen)
    
    # filter out non-alpha words
    for word in words:
       if re.search(u'[a-zA-Z]', word):
           if len(word) > 3:
               filtered_words.append(word.lower())
           
    # stemming
    for w in filtered_words:
        stems.append(stemmer.stem(w))
    
    if len(stems) == 0:
        stems = ['sinnombr']

    return ' '.join(stems)

# Process names
df_client['client_stems'] = np.array
df_client['client_stems'] = df_client.apply(lambda row: word_tokenize_stem(row),axis=1)


# Vectorizer
#vectorizer = CountVectorizer(min_df=1)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_client.client_stems)

# Clustering
#cls = KMeans(n_clusters = 64)
#km = cls.fit_predict(X)
#df_client['client_tfid_64'] = km.astype(np.int32)
#
#cls = KMeans(n_clusters = 512)
#km = cls.fit_predict(X)
#df_client['client_tfid_512'] = km.astype(np.int32)

## Get mean demand values by client from df_full
#df_full = pd.read_csv("input/train.csv")
#df_full.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
#                   'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
#                   'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
#by_mean = df_full.groupby('ClientId').agg({'AdjDemand':np.mean}).reset_index()
#by_mean = by_mean.rename(columns = {'AdjDemand':'client_mean_demand'})
#df_client = pd.merge(df_client,by_mean,on=['ClientId'],how='left')


#store = pd.HDFStore('processed/processed_data.h5',complevel=9, complib='bzip2')
#store['clients'] = df_client
#store.close()
