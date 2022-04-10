import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field

# Pythonは動的型付け言語で「型ヒント（Type Hint）」を使って関数のシグネチャなどに型を付与することが出来ます。
# 通常、コードの中身には何も作用しないが、Pydantic という強力なライブラリによって
# 型ヒントを基にAPIの入出力のバリデーションを行います。

# -------------------------------
# pydantic でデータバリデーション（型の定義/スキーマの定義）
# -------------------------------

class Result(BaseModel):
    # result: dict = Field(None, description="推論結果。結果はdfになっている", example=" 'result_message': 'Predicted data inserted','Predicted_data': df ")
    result_message: str
    # pandas のバリデーションは難しいのでなし
    # Predicted_data: pd.DataFrame = Field(None, description="推論結果。結果はdfになっている")

# Optionalは、指定の型 or Noneを、許容する型です。
# 例えば str と None のいずれかを許容する場合は Optional[str] とします。
# Field で色々情報を付与
# class Path(BaseModel):
#     path: Optional[str] = Field(None, description="データがあるパス", example="./file/path")

