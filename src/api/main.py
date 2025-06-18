from fastapi import FastAPI
from auth_api import router as auth_router
from predict_api import router as predict_router
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Inclure les sous-applications
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(predict_router, prefix="/protected", tags=["predict"])

@app.get("/")
def verify_api():
    return {"message": "Bienvenue ! L'API est fonctionnelle."}

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
