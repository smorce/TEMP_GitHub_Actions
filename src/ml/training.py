import numpy as np
import pandas as pd
import pickle
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 実行上問題ないwarningは非表示にする
import warnings
warnings.filterwarnings('ignore')


ON_RETRAIN_MODE = os.environ.get('ON_RETRAIN_MODE')


# **BigQueryからデータをロードする**

# In[7]:


from google.cloud import bigquery

bqclient = bigquery.Client()

# Download query results.
# ===================================================
# 全データをロードして df に保存する
# 超時間がかかるのでランダムサンプルしてLIMITをかける
# トレーニングデータは noise = 10 のデータに絞る
# ===================================================
if ON_RETRAIN_MODE :
    # 再学習フロー
    # 違い：こっちは最新のデータから1500件持ってくる
    query_string = """
    SELECT
        y
        ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
        ,_airbyte_emitted_at
    FROM
        df_on_missing_value_completion.df_on_missing_value_completion
    WHERE
        noise = 10
    ORDER BY
        _airbyte_emitted_at DESC
    LIMIT
        1500    # 3000にすると20分以上かかる。デバッグ中だけ100にしてた
    # GROUP BY
    #     y
    #     ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
    """
else :
    # 通常のトレーニングフロー
    # ランダムに1500件持ってきて学習させる
    query_string = """
    SELECT
        y
        ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
        ,rand() AS random
        # ,MAX(_airbyte_emitted_at) AS _airbyte_emitted_at
    FROM
        df_on_missing_value_completion.df_on_missing_value_completion
    WHERE
        noise = 10
    ORDER BY
        random
    LIMIT
        1500    # 3000にすると20分以上かかる
    # GROUP BY
    #     y
    #     ,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
    """
print("!----- データ集計中… -----!")
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
print("!----- 集計が完了しました -----!")

# del df['_airbyte_emitted_at']


# In[25]:



# # ガウス過程回帰 5Fold クロスバリデーション
#  - 最適なカーネルはデータセットによって異なるため、グリッドサーチで最適なカーネルを探索している
#  - ガウス過程回帰モデルには「特徴量と目的変数を標準化した」「array型」のデータセットを渡す

# In[37]:

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.model_selection import train_test_split, GridSearchCV

K_FOLD = 5

# dataset(array型)
y = df.y.values


print('デバッグ中')
print('yの合計値 =', y.sum())


if ON_RETRAIN_MODE :
    x = df.drop(columns=['y','_airbyte_emitted_at']).values
else:
    x = df.drop(columns=['y','random']).values

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

print("!----- トレーニングを開始します… -----!")

# Gaussian process regression
cv_model = GridSearchCV(GaussianProcessRegressor(alpha=0), {'kernel': kernels}, cv=K_FOLD)
cv_model.fit(autoscaled_x_train, autoscaled_y_train)
optimal_kernel = cv_model.best_params_['kernel']
model = GaussianProcessRegressor(optimal_kernel, alpha=0)
model.fit(autoscaled_x_train, autoscaled_y_train)

print("!----- トレーニングが完了しました -----!")

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
filename = "./models/GaussianProcessRegressor.pkl"
pickle.dump(model, open(filename, 'wb'))

# 初回トレーニングに使用したデータの保存
# ドリフトチェックで必要
df.to_csv("./data/training.csv", index=False)


# 予測するときに標準化したものを割り戻さないといけないので平均値と標準偏差も保存する
# ドリフトチェックのために学習時の平均値と標準偏差も保存する

_dict = {'y_mean' : y_train.mean(), 'y_std' : y_train.std(ddof=1), 'predicted_y_test' : predicted_y_test, 'predicted_y_test_std' : predicted_y_test_std}
pickle.dump(_dict, open("./data/mean_and_std.txt", "wb") )


