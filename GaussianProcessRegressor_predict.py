#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 実行上問題ないwarningは非表示にする
import warnings
warnings.filterwarnings('ignore')



# **BigQueryからデータをロードする**

# In[4]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# 下記で最新の1000件をとってくる

# In[5]:


get_ipython().run_cell_magic('bigquery', 'df', '# ===================================================\n# 最新のデータをロードして df に保存する\n# ===================================================\nSELECT\n    y\n    ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10\n    ,MAX(_airbyte_emitted_at) AS _airbyte_emitted_at\nFROM\n    df_on_missing_value_completion.df_on_missing_value_completion\nGROUP BY\n    y\n    ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10')


# In[6]:


del df['_airbyte_emitted_at']


# In[7]:


df


# # ガウス過程回帰モデルを読み込む

# In[8]:

# 構築したモデルの読み込み
filename = './GaussianProcessRegressor.pkl'
gpr = pickle.load(open(filename,  'rb'))


# # 予測に必要な平均値と標準偏差を読み込む

# In[9]:


filename = './mean_and_std.txt'
_dict = pickle.load(open(filename, 'rb'))



# # y を予測する

# In[12]:

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
import matplotlib.figure as figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# dataset(array型)
# y = df.y.values
x = df.drop(columns='y').values

# autoscaling(標準化)
autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
# autoscaled_y = (y - y.mean()) / y.std(ddof=1)

# prediction
# 標準化しているので、標準化したときの計算の逆をやることで割り戻している。標準偏差の方は y_train の標準偏差をかけるだけで散らばり具合が戻せる
predicted_y_test, predicted_y_test_std = gpr.predict(autoscaled_x, return_std=True)
predicted_y_test = predicted_y_test * _dict['y_std'] + _dict['y_mean']
predicted_y_test_std = predicted_y_test_std * _dict['y_std']

# In[25]:





