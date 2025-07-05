from fastapi import FastAPI
from predict_api import router as predict_api
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()

app.include_router(predict_api, prefix="/protected", tags=["predict"])

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
