import sys
from fastapi import FastAPI
# routers を読み込むために相対パスを追加
sys.path.append('src/api')
import routers

# Swagger UI に表示するため、作成した router インスタンスを、FastAPIインスタンスに取り込む
app = FastAPI()
app.include_router(routers.router)