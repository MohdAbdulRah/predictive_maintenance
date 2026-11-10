"""uvicorn api.main:app --reload --port 8000"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import machine, maintenance, model, predictions
from database.connection import get_engine

@asynccontextmanager
async def lifespan(_):
    get_engine()
    yield


app = FastAPI(title="MTConnect Predictive Maintenance API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
                   allow_methods=["GET"], allow_headers=["*"])
for r in (machine.router, predictions.router, maintenance.router, model.router):
    app.include_router(r)


@app.get("/health")
def health():
    return {"ok": True}
