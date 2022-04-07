#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 実行上問題ないwarningは非表示にする
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/df.csv')


# **欠損値補完（MICE）**

# In[2]:


from statsmodels.imputation import mice

# 辞書を作成
dict_mice = {}
for col, i in zip(df.columns, df.isnull().sum()):
    dict_mice[col] = i


# In[3]:


# 値ソート（小さい順から）
dict_mice_sorted = sorted(dict_mice.items(), key=lambda x:x[1])


# In[4]:


# 全colに処理
list_columns_temp = []
for col, i in dict_mice_sorted:
    # tempに追加
    list_columns_temp.append(col)
    # 処理が必要な場合はmiceを実施
    if i > 0:
        mice_imp = mice.MICEData(df[list_columns_temp])
        mice_imp.set_imputer(col, formula=" + ".join(list_columns_temp))
        mice_imp.update_all()
        # 適用
        df[list_columns_temp] = mice_imp.data
    else:
        pass


# In[10]:


df.to_csv('./data/df_on_missing_value_completion.csv', index=False)


# **作成した欠損データを GCS にアップロードする**

# In[11]:


# **CloudStorageに接続**

# In[6]:

from google.cloud import storage
client = storage.Client()
# https://console.cloud.google.com/storage/browser/[bucket-id]/
bucket = client.get_bucket('mlops_1')


# **CloudStorageにファイルをアップロードする**

# In[11]:


# 保存ファイル名（フォルダを指定することもできる）
save_file_name = 'df_on_missing_value_completion.csv'
# アップロードしたいファイルのパス
uploaded_file_path = './data/df_on_missing_value_completion.csv'
blob = bucket.blob(save_file_name)
blob.upload_from_filename(filename=uploaded_file_path)


# In[ ]:

