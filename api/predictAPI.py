#!/usr/bin/env python
# coding: utf-8

# In[1]:

from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 実行上問題ないwarningは非表示にする
import warnings
warnings.filterwarnings('ignore')



# # ガウス過程回帰モデルを読み込む

# In[8]:

# 構築したモデルの読み込み
filename = './GaussianProcessRegressor.pkl'
gpr = pickle.load(open(filename,  'rb'))


# 予測に必要な平均値と標準偏差を読み込む
filename = './mean_and_std.txt'
_dict = pickle.load(open(filename, 'rb'))


# 学習で使ったデータを読み込む
filename = './data/training.csv'
training = pd.read_csv(filename)
# 学習時の最新の日付
last_training_date = training._airbyte_emitted_at.max()


# **予測するためBigQueryからユーザのログデータをロードする**

from google.cloud import bigquery

bqclient = bigquery.Client()

# Download query results.
# ===================================================
# 最新のデータをロードして df に保存する
# ===================================================
query_string = """
SELECT
    y
    ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
    ,MAX(_airbyte_emitted_at) AS _airbyte_emitted_at
FROM
    df_on_missing_value_completion.df_on_missing_value_completion
GROUP BY
    y
    ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
"""

df = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # create_bqstorage_client はデフォルトで True
        # BigQuery Storage APIは、BigQueryから行をフェッチするためのより高速な方法で、True で使用する
        # https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.job.QueryJob.html
        create_bqstorage_client=True,
    )
)

# 予測時の日付データは消さない。どのタイミングのデータを使って予測したのかを記録する
# del df['_airbyte_emitted_at']


# **予測したデータもロードする**
# 欲しいのは最新の日付だけ(last_prediction)
# query_string = """
# SELECT
#     MAX(_airbyte_emitted_at) AS last_prediction
# FROM
#     df_on_missing_value_completion.predicted_df_on_missing_value_completion
# """
# predicted = (
#             bqclient.query(query_string)
#             .result()
#             .to_dataframe(
#                 # Optionally, explicitly request to use the BigQuery Storage API. As of
#                 # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
#                 # create_bqstorage_client はデフォルトで True
#                 # BigQuery Storage APIは、BigQueryから行をフェッチするためのより高速な方法で、True で使用する
#                 # https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.job.QueryJob.html
#                 create_bqstorage_client=True,
#     )
# )

# last_prediction_date = predicted['last_prediction']


# ここまでで準備は整ったのでFastAPIをインスタンス化する
# Initialize an instance of FastAPI
app = FastAPI()

# Define the route to the predictor
# y を予測する
# POST: データの作成
@app.post("/gpr_predict")
def gpr_predict():

    # セッティング
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


    # 標準化しているので、標準化したときの計算の逆をやることで割り戻している。標準偏差の方は y_train の標準偏差をかけるだけで散らばり具合が戻せる
    predicted_y_test, predicted_y_test_std = gpr.predict(autoscaled_x, return_std=True)
    predicted_y_test = predicted_y_test * _dict['y_std'] + _dict['y_mean']
    predicted_y_test_std = predicted_y_test_std * _dict['y_std']


    # In[25]:

    # # グラフ描画と保存
    index = df.index

    plt.rcParams['font.size'] = 12  # 横軸や縦軸の名前の文字などのフォントのサイズ

    plt.figure(figsize=(10, 5))
    plt.plot(index, predicted_y_test, color='navy', label='Predicted Mean')
    plt.fill_between(index, predicted_y_test-3.00*predicted_y_test_std, predicted_y_test+3.00*predicted_y_test_std, color='navy', alpha=0.2, label='Predicted Boundaries - 3sigma')
    plt.xlabel('Index Datapoint')
    plt.ylabel('Predicted Mean')
    plt.legend()

    plt.savefig('./artifact/Predicted_Mean.png', dpi=100, bbox_inches="tight")
    plt.clf() # plt.clf() → plt.close() でメモリ解放
    plt.close('all')


    # # 入力データと予測結果をBigQueryに書き込む
    # pandas.to_gbq のやり方しか見つからなかった
    import os
    project_id = os.environ.get('project_id')

    # df に予測値を入れる
    df['GPR_Mean_Predicted'] = predicted_y_test
    df['GPR_Upper_Boundary_on3sigma']= predicted_y_test+3.00*predicted_y_test_std
    df['GPR_Lower_Boundary_on3sigma']= predicted_y_test-3.00*predicted_y_test_std


    outlier = []

    for i in range(len(df)):
        o = "Non Outlier"
        if df['y'].loc[i]<=df['GPR_Upper_Boundary_on3sigma'].loc[i] and df['y'].loc[i]>=df['GPR_Lower_Boundary_on3sigma'].loc[i]:
            True
        else:
            o="Outlier"
        outlier.append(o)

    df['Outlier']=outlier

    outlier_spec = []

    for i in range(len(df)):
        o = "Non Outlier"
        if df['y'].loc[i]>=df['GPR_Upper_Boundary_on3sigma'].loc[i]:
            o = "Upper Outlier"
        if df['y'].loc[i]<=df['GPR_Lower_Boundary_on3sigma'].loc[i]:
            o = "Lower Outlier"
        else:
            True
        outlier_spec.append(o)

    df['Outlier_Type'] = outlier_spec

    # if_exists="replace" : 同じものがあったら上書き保存する
    df.to_gbq("df_on_missing_value_completion.predicted_df_on_missing_value_completion", project_id=project_id, if_exists="replace")

    return {
            "result_message": 'Predicted data inserted',
            "Predicted_data": df
            }


