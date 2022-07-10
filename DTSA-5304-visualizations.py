#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import json
import numpy as np
import pandas as pd
from pandas import json_normalize

import datetime
import altair as alt

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[22]:


def load_df(csv_path, nrows = None):
    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path,
                     converters = {col: json.loads for col in json_cols},                                                                         
                         dtype = {'fullVisitorId': 'str'}, # Important!!
                         nrows = nrows)
    for col in json_cols:
        flat_col = json_normalize(df[col])
        flat_col.columns = [f"{col}_{subcol}" for subcol in flat_col.columns]
        df = df.drop(col, axis = 1).merge(flat_col, right_index = True, left_index = True)
    return df


# In[23]:


# data = https://www.kaggle.com/competitions/ga-customer-revenue-prediction/data?select=train.csv

train = load_df("train.csv", nrows = 5000)
train.head()


# In[24]:


train.info()


# In[37]:


total =["totals_visits", "totals_hits" , "totals_pageviews" , "totals_bounces" , "totals_newVisits" , "totals_transactionRevenue" ]
for t in total:
    train[t] = pd.to_numeric(train[t])
train.info()


# In[44]:


train_buy = train[train['totals_transactionRevenue'] > 0]
train_no_buy = train[train['totals_transactionRevenue'].isna()]


# In[34]:


alt.Chart(train).mark_bar().encode(alt.X('channelGrouping'), alt.Y('count()', title='Count'))


# In[45]:


alt.Chart(train_buy).mark_bar().encode(alt.X('channelGrouping'), alt.Y('count()', title='Count'))


# In[46]:


alt.Chart(train_no_buy).mark_bar().encode(alt.X('channelGrouping'), alt.Y('count()', title='Count'))


# In[35]:


alt.Chart(train).mark_bar().encode(alt.X('device_browser') ,y='count()')


# In[47]:


alt.Chart(train_buy).mark_bar().encode(alt.X('device_browser') ,y='count()')


# In[48]:


alt.Chart(train_no_buy).mark_bar().encode(alt.X('device_browser') ,y='count()')


# In[42]:


alt.Chart(train).mark_bar().encode(alt.X('device_operatingSystem') ,y='count()')


# In[49]:


alt.Chart(train_buy).mark_bar().encode(alt.X('device_operatingSystem') ,y='count()')


# In[50]:


alt.Chart(train_no_buy).mark_bar().encode(alt.X('device_operatingSystem') ,y='count()')


# In[53]:


alt.Chart(train_buy).mark_bar().encode(alt.X('trafficSource_source') ,y='count()')


# In[54]:


alt.Chart(train_no_buy).mark_bar().encode(alt.X('trafficSource_source') ,y='count()')


# In[58]:


alt.Chart(train_buy).mark_point().encode(
    x='totals_pageviews',
    y='count()',
    color='device_isMobile'
)


# In[59]:


alt.Chart(train_no_buy).mark_point().encode(
    x='totals_pageviews',
    y='count()',
    color='device_isMobile'
)


# In[62]:


alt.Chart(train_buy).mark_bar().encode(alt.X('geoNetwork_country') ,y='count()')


# In[63]:


alt.Chart(train_buy).mark_bar().encode(alt.X('geoNetwork_city') ,y='count()')


# In[60]:


alt.Chart(train_buy).mark_point().encode(
    x='totals_visits',
    y='count()',
    color='device_isMobile'
)


# In[61]:


alt.Chart(train_no_buy).mark_point().encode(
    x='totals_visits',
    y='count()',
    color='device_isMobile'
)


# In[ ]:




