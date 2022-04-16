import sys
from fastapi import APIRouter, HTTPException

import schemas as schema
# predict を読み込むために相対パスを追加
sys.path.append('src/ml')
# from src.ml.predict では読み込めなかった
from predict_model import Model

router = APIRouter()

# -------------------------------
# routersのプレースホルダ
# -------------------------------

"""
[!- schemas.py -!]
# Optionalは、指定の型 or Noneを、許容する型です。
# 例えば str と None のいずれかを許容する場合は Optional[str] とします。
# Field で色々情報を付与
class Task(BaseModel):　　←　下記の schema.Task がこれ
    id: int
    title: Optional[str] = Field(None, example="クリーニングを取りに行く")
    done: bool = Field(False, description="完了フラグ")
---
# レスポンスのスキーマは response_model にセットする。複数返すのでList型にしている
@router.get("/tasks", response_model=List[schema.Task])
async def list_tasks():
    return [schema.Task(id=1, title="1つ目のTODOタスク")]
"""
@router.get("/")
async def hello():
    return {"message": "hello world!"}

# GET: データの取得
@router.get("/health")
async def health():
    """
    Web API の実行確認
    """
    return {"health": "ok"}

# POST: データの作成
# response_model を定義すると勝手に辞書の内容を展開して、そのクラスの変数にアンパック代入される
# 下記の例だと Result クラスの変数に勝手にアンパック代入されるため、分かりやすくするために明示する
@router.post("/predict/batch", response_model=schema.Result, status_code=200)
# async def predict(path: schema.Path):
async def predict():
    """
    BigQuery のデータを読み込み推論する
    :model.df: pandasデータフレーム形式
    """
    # Modelクラスのインスタンス作成
    # model = Model(path=path)
    print("!----- モデルのインスタンスを作成します -----!")
    model = Model()
    print("!----- インスタンスの作成が完了しました -----!")
    print("!----- 予測を開始します -----!")
    model.predict()
    print("!----- 予測が完了しました -----!")
    print("!----- グラフの描画と保存をします -----!")
    model.make_save_figure()
    print("!----- グラフの描画と保存が完了しました -----!")
    print("!----- 予測結果をBigQueryにアップロードします -----!")
    model.insert()
    print("!----- アップロードが完了しました -----!")

    # Outlier_Type の値が何もなければエラーを返す
    if model.df['Outlier_Type'].max() == None:
        raise HTTPException(status_code=400, detail="Results not found.")
    else:
        result_dict = {
        "result_message": 'Predicted data inserted',
        "Predicted_data": model.df
        }
        # return result_dict　←　この書き方でも良いが分かりづらいのでやめた
        return schema.Result(**result_dict)