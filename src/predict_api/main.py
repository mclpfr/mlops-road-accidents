from fastapi import FastAPI
from predict_api import router as predict_api

app = FastAPI()

app.include_router(predict_api, prefix="/predict", tags=["predict"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
