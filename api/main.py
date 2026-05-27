"""
FastAPI 서버 v2 - 서울랜드 입장객 예측 (기상청 데이터 통합 모델)

실행: uvicorn api.main:app --reload --port 8000
문서: http://localhost:8000/docs
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
from predict import predict_visitors

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="서울랜드 입장객 예측 API (v5b-final)",
    description="6개 데이터셋(서울랜드 + 기상청) + XGBoost/LightGBM 앙상블. R² = 0.798, MAPE = 37.8%",
    version="5.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    date: str = Field(..., example="2026-05-05")
    weather: Literal["맑음", "흐림", "비", "눈", "안개", "기타"] = "맑음"
    min_temp: float = Field(15, example=15)
    max_temp: float = Field(22, example=22)
    precipitation_mm: float = Field(0, example=0, description="강수량(mm)")
    operating_hours: float = Field(11.0, example=11.0)
    avg_temp: Optional[float] = Field(None, description="없으면 (min+max)/2")


class PredictResponse(BaseModel):
    date: str
    predicted_visitors: int
    average_visitors: int
    congestion_ratio: float
    congestion_percent: int
    congestion_level: str
    congestion_emoji: str
    message: str
    recommendation: str


try:
    with open(ROOT / "data" / "stats.json", encoding="utf-8") as f:
        STATS = json.load(f)
except FileNotFoundError:
    STATS = {}


@app.get("/")
def root():
    # 실제 사용 중인 모델을 동적으로 반영 (앙상블/단일 자동 인지)
    from predict import _USE_ENSEMBLE
    try:
        import joblib as _j
        _m = _j.load(ROOT / "models" / "metadata.pkl")
        return {
            "status": "ok", "service": "seoulland-ml", "version": _m.get("version", "v5"),
            "model": _m.get("model_name", "XGBoost"),
            "mode": "ensemble" if _USE_ENSEMBLE else "single-xgb-fallback",
            "model_r2": round(_m.get("test_r2", 0), 4),
            "model_mae": round(_m.get("test_mae", 0)),
            "model_mape": round(_m.get("test_mape", 0), 2),
            "docs": "/docs",
        }
    except Exception:
        return {"status": "ok", "service": "seoulland-ml", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/stats")
def get_stats():
    return STATS


@app.post("/predict", response_model=PredictResponse)
def predict_post(req: PredictRequest):
    try:
        return predict_visitors(
            date=req.date, weather=req.weather,
            min_temp=req.min_temp, max_temp=req.max_temp,
            precipitation_mm=req.precipitation_mm,
            operating_hours=req.operating_hours,
            avg_temp=req.avg_temp,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict", response_model=PredictResponse)
def predict_get(
    date: str = Query(..., example="2026-05-05"),
    weather: Literal["맑음", "흐림", "비", "눈", "안개", "기타"] = Query("맑음"),
    min_temp: Optional[float] = Query(None),
    max_temp: Optional[float] = Query(None),
    precipitation_mm: float = Query(0.0),
    operating_hours: float = Query(11.0),
):
    """간단 호출 — 기온 생략 시 월별 평년값 자동 사용."""
    month_temp = {
        1: (-3, 4), 2: (-1, 7), 3: (3, 11), 4: (9, 18), 5: (14, 23), 6: (19, 28),
        7: (23, 30), 8: (23, 31), 9: (18, 27), 10: (10, 20), 11: (4, 13), 12: (-2, 6),
    }
    try:
        import pandas as pd
        month = pd.Timestamp(date).month
        lo = min_temp if min_temp is not None else month_temp[month][0]
        hi = max_temp if max_temp is not None else month_temp[month][1]
        return predict_visitors(
            date=date, weather=weather,
            min_temp=lo, max_temp=hi,
            precipitation_mm=precipitation_mm,
            operating_hours=operating_hours,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
