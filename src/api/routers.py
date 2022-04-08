from fastapi import APIRouter, HTTPException

from src.api import schemas as schema

from src.ml.predict import Model

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
    model = Model()

    model.predict()
    model.make_save_figure()
    model.insert()

    if model.df['Outlier_Type'] == None:
        raise HTTPException(status_code=400, detail="Results not found.")
    else:
        result_dict = {
        "result_message": 'Predicted data inserted',
        "Predicted_data": model.df
        }
        # return result_dict　←　この書き方でも良い
        return schema.Result(**result_dict)