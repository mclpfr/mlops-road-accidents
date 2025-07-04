from fastapi import FastAPI
from predict_routes import router as predict_router

app = FastAPI()

app.include_router(predict_router, prefix="/predict", tags=["predict"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
