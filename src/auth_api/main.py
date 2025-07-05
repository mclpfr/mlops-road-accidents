from fastapi import FastAPI
from auth_api import router as auth_api
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()

app.include_router(auth_api, prefix="/auth", tags=["auth"])

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7999)
