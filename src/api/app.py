from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.api.feature_builder import build_features_for_point
from src.api.predict import predict_for_point

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PointRequest(BaseModel):
    lat: float
    lon: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/features")
def features(point: PointRequest):
    try:
        result = build_features_for_point(point.lat, point.lon)
        return {"success": True, "features": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/predict")
def predict(point: PointRequest):
    try:
        result = predict_for_point(point.lat, point.lon)
        return {
            "success": True,
            **result,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
