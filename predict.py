"""
predict.py - v2 모델 (서울랜드 + 기상청 데이터 통합 학습)

사용:
    from predict import predict_visitors
    r = predict_visitors(
        date="2026-08-15", weather="맑음",
        min_temp=22, max_temp=31,
        precipitation_mm=0,           # 강수량 mm (없으면 0)
        avg_temp=None,                 # 미지정 시 (min+max)/2 자동
        operating_hours=12,
    )
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import holidays as kr_holidays_mod

ROOT = Path(__file__).resolve().parent
_meta = joblib.load(ROOT / "models" / "metadata.pkl")
_FEATURES = _meta["features"]
_OVERALL_MEAN = _meta["overall_mean"]

# 앙상블 모델이 있으면 사용, 없으면 단일 모델 폴백
_ens_path = ROOT / "models" / "model_ensemble.pkl"
if _ens_path.exists():
    _ens = joblib.load(_ens_path)
    _xgb_model, _lgb_model = _ens["xgb"], _ens["lgb"]
    _xgb_weight = _ens.get("weight_xgb", 0.5)
    _USE_ENSEMBLE = True
    print(f"[predict] Ensemble loaded: XGB({_xgb_weight:.2f}) + LGB({1-_xgb_weight:.2f})")
else:
    _model = joblib.load(ROOT / "models" / "model.pkl")
    _USE_ENSEMBLE = False
    print(f"[predict] Single model loaded")


def _model_predict(X):
    """앙상블이면 가중평균, 아니면 단일 모델 사용."""
    if _USE_ENSEMBLE:
        xpred = np.expm1(_xgb_model.predict(X))
        lpred = np.expm1(_lgb_model.predict(X))
        return _xgb_weight * xpred + (1 - _xgb_weight) * lpred
    else:
        return np.expm1(_model.predict(X))

_kr_holidays = kr_holidays_mod.country_holidays("KR", years=range(2020, 2031))

def _is_holiday(d):
    """holidays 라이브러리 + 근로자의날(5/1) 보강."""
    if d.date() in _kr_holidays:
        return True
    if d.month == 5 and d.day == 1:  # 근로자의날
        return True
    return False

def _is_sandwich(d):
    """공휴일과 인접한 평일 (전날/다음날)."""
    if d.dayofweek >= 5 or _is_holiday(d):
        return False
    yesterday = d - pd.Timedelta(days=1)
    tomorrow = d + pd.Timedelta(days=1)
    return _is_holiday(yesterday) or _is_holiday(tomorrow)

# 월별 일조시간 디폴트 (서울 평년)
_DAYLIGHT_DEFAULT = {
    1: 10.0, 2: 10.8, 3: 11.9, 4: 13.1, 5: 14.0, 6: 14.5,
    7: 14.3, 8: 13.5, 9: 12.3, 10: 11.1, 11: 10.2, 12: 9.7,
}


def _build_row(date, weather, min_temp, max_temp, operating_hours,
               precipitation_mm, avg_temp, daylight_hours):
    d = pd.Timestamp(date)

    # 디폴트 처리
    if avg_temp is None:
        avg_temp = (min_temp + max_temp) / 2
    if daylight_hours is None:
        daylight_hours = _DAYLIGHT_DEFAULT.get(d.month, 11.5)

    # 기상청 카테고리는 입력 날씨에서 추정 (가장 유사한 것 매핑)
    # 사용자 입력은 5종(맑음/흐림/비/눈/기타) → KMA 6종(맑음/흐림/비/눈/안개/기타)
    weather_kma = weather  # 동일 매핑 (안개는 입력 시에만 별도 처리)

    row = {
        "월": d.month, "일": d.day, "요일": d.dayofweek,
        "주말여부": 1 if d.dayofweek >= 5 else 0,
        "공휴일": 1 if _is_holiday(d) else 0,
        "방학": 1 if d.month in (1, 2, 7, 8) else 0,
        "샌드위치데이": 1 if _is_sandwich(d) else 0,
        "평균기온": avg_temp,
        "최저기온": min_temp,
        "최고기온": max_temp,
        "일강수량": precipitation_mm,
        "강수여부": 1 if precipitation_mm > 0 else 0,
        "폭우": 1 if precipitation_mm > 30 else 0,
        "일조시간_시간": daylight_hours,
        "운영시간_시간": operating_hours,
    }
    # 두 종류 날씨 원-핫
    for col in _FEATURES:
        if col.startswith("날씨내부_"):
            row[col] = 1 if col == f"날씨내부_{weather}" else 0
        elif col.startswith("날씨기상청_"):
            row[col] = 1 if col == f"날씨기상청_{weather_kma}" else 0

    # 학습 때와 동일한 순서
    return pd.DataFrame([[row.get(f, 0) for f in _FEATURES]], columns=_FEATURES)


def predict_visitors(date, weather="맑음", min_temp=15.0, max_temp=22.0,
                     operating_hours=11.0, precipitation_mm=0.0,
                     avg_temp=None, daylight_hours=None):
    """
    예상 입장객 + 4단계 혼잡도 반환.

    필수: date
    선택: weather, min_temp, max_temp, precipitation_mm 등 — 없으면 합리적 디폴트
    """
    X = _build_row(date, weather, min_temp, max_temp, operating_hours,
                   precipitation_mm, avg_temp, daylight_hours)
    pred = max(float(_model_predict(X)[0]), 0)

    ratio = pred / _OVERALL_MEAN
    pct = int(round((ratio - 1) * 100))

    # 4단계 분류
    if pct <= -50:
        level, emoji = "한산", "🟢"
    elif pct < 20:
        level, emoji = "보통", "🟡"
    elif pct < 50:
        level, emoji = "붐빔", "🟠"
    else:
        level, emoji = "매우 붐빔", "🔴"

    if pct > 5:
        msg = f"평균보다 {pct}% 더 붐빌 예정"
    elif pct < -5:
        msg = f"평균보다 {abs(pct)}% 덜 붐빌 예정"
    else:
        msg = "평균 수준의 혼잡도"

    # 추천 메시지
    d = pd.Timestamp(date)
    factors = []
    if _is_holiday(d): factors.append("공휴일")
    if _is_sandwich(d): factors.append("샌드위치데이")
    if d.dayofweek >= 5: factors.append("주말")
    if d.month in (7, 8): factors.append("여름 성수기")
    elif d.month in (4, 5, 10): factors.append("나들이 좋은 계절")
    if precipitation_mm > 30: factors.append("폭우")
    elif precipitation_mm > 0: factors.append("우천")
    if weather == "눈": factors.append("강설")

    factor_str = " + ".join(factors)
    if level == "매우 붐빔":
        rec = f"매우 붐빌 예정입니다. {factor_str} 영향." if factors else "매우 붐빌 예정입니다."
    elif level == "붐빔":
        rec = f"평소보다 붐빌 예정입니다. {factor_str} 영향." if factors else "평소보다 붐빌 예정입니다."
    elif level == "한산":
        rec = f"여유롭게 즐기기 좋은 날입니다. {factor_str} 영향." if factors else "여유롭게 즐기기 좋은 날입니다."
    else:
        rec = "평소 수준의 방문이 예상됩니다."

    return {
        "date": date,
        "predicted_visitors": int(round(pred)),
        "average_visitors": int(round(_OVERALL_MEAN)),
        "congestion_ratio": round(ratio, 2),
        "congestion_percent": pct,
        "congestion_level": level,
        "congestion_emoji": emoji,
        "message": msg,
        "recommendation": rec,
    }


if __name__ == "__main__":
    examples = [
        ("2026-05-05", "맑음", 15, 24, 12, 0),    # 어린이날
        ("2026-08-15", "맑음", 22, 31, 12, 0),    # 광복절
        ("2026-11-25", "비",   5, 12, 10, 15),    # 평일 우천
        ("2026-12-25", "눈",  -5,  2, 10, 5),     # 크리스마스
        ("2026-07-20", "비",  23, 28, 11, 80),    # 폭우 평일
    ]
    for date, w, lo, hi, oh, rain in examples:
        r = predict_visitors(date, w, lo, hi, oh, rain)
        print(f"{r['date']} {w}({rain}mm): {r['predicted_visitors']:>6,}명  "
              f"{r['congestion_emoji']} {r['congestion_level']:5s}  "
              f"{r['message']}")
