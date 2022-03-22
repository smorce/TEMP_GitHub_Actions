#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 実行上問題ないwarningは非表示にする
import warnings
warnings.filterwarnings('ignore')



# **BigQueryからデータをロードする**

# In[7]:

from IPython import get_ipython
ipython = get_ipython()

%load_ext google.cloud.bigquery

ipython.magic('load_ext', 'google.cloud.bigquery')
# get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# 下記で最新の1000件をとってこれる。すべてを学習データにしたい場合はMAXとGROUPを外す。<br>
# 一旦、最新のデータを使って学習する。

# In[20]:


get_ipython().run_cell_magic('bigquery', 'df', '# ===================================================\n# 最新のデータをロードして df に保存する\n# ===================================================\nSELECT\n    y\n    ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10\n    ,MAX(_airbyte_emitted_at) AS _airbyte_emitted_at\nFROM\n    df_on_missing_value_completion.df_on_missing_value_completion\nGROUP BY\n    y\n    ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10')


# In[24]:


del df['_airbyte_emitted_at']


# In[25]:



# # ガウス過程回帰 5Fold クロスバリデーション
#  - 最適なカーネルはデータセットによって異なるため、グリッドサーチで最適なカーネルを探索している
#  - ガウス過程回帰モデルには「特徴量と目的変数を標準化した」「array型」のデータセットを渡す

# In[37]:

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.figure as figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')

K_FOLD = 5

# dataset(array型)
y = df.y.values
x = df.drop(columns='y').values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# グリッドサーチで最適化するカーネル群
kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
               ConstantKernel() * RBF() + WhiteKernel(),  # これがちょうど良い塩梅でオススメらしい
               ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
               ConstantKernel() * RBF(np.ones(x_train.shape[1])) + WhiteKernel(),
               ConstantKernel() * RBF(np.ones(x_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
               ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
               ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
               ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
               ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
               ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
               ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]

# autoscaling(標準化)
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# Gaussian process regression
cv_model = GridSearchCV(GaussianProcessRegressor(alpha=0), {'kernel': kernels}, cv=K_FOLD)
cv_model.fit(autoscaled_x_train, autoscaled_y_train)
optimal_kernel = cv_model.best_params_['kernel']
model = GaussianProcessRegressor(optimal_kernel, alpha=0)
model.fit(autoscaled_x_train, autoscaled_y_train)

# calculate y in training data
calculated_y_train = model.predict(autoscaled_x_train) * y_train.std(ddof=1) + y_train.mean()

# prediction
# 標準化しているので、標準化したときの計算の逆をやることで割り戻している。標準偏差の方は y_train の標準偏差をかけるだけで散らばり具合が戻せる
predicted_y_test, predicted_y_test_std = model.predict(autoscaled_x_test, return_std=True)
predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
predicted_y_test_std = predicted_y_test_std * y_train.std(ddof=1)


# # 学習したモデルをpickleで保存する

# In[39]:

# 構築したモデルの保存
filename = './GaussianProcessRegressor.pkl'
pickle.dump(model, open(filename, 'wb'))


# # 予測するときに標準化したものを割り戻さないといけないので平均値と標準偏差も保存する

# In[40]:


_dict = {'y_mean' : y_train.mean(), 'y_std' : y_train.std(ddof=1)}
pickle.dump(_dict, open("./mean_and_std.txt", "wb") )


# In[ ]:




