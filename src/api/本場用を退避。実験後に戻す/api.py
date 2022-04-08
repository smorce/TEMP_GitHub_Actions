from fastapi import FastAPI
from src.api import routers

# Swagger UI に表示するため、作成した router インスタンスを、FastAPIインスタンスに取り込む
app = FastAPI()
app.include_router(routers.router)