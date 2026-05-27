"""
04_compare_models.py - Random Forest vs XGBoost 성능 비교

같은 데이터로 두 모델을 학습한 뒤, 2025년 검증 데이터에서
누가 더 정확한지 비교하고 더 나은 모델을 최종 모델로 저장합니다.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean.csv", parse_dates=["날짜"])

weather_dummies = pd.get_dummies(df["날씨"], prefix="날씨")
df = pd.concat([df, weather_dummies], axis=1)

FEATURES = (["월", "일", "요일", "주말여부", "공휴일", "방학",
             "최저기온", "최고기온", "운영시간_시간"]
            + list(weather_dummies.columns))
TARGET = "실입장객"

train = df[df["연도"] < 2025].copy()
test = df[df["연도"] == 2025].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
y_train_log = np.log1p(y_train)

# ===== Random Forest =====
print("🌳 Random Forest 학습 중...")
rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_log)
rf_pred = np.clip(np.expm1(rf.predict(X_test)), 0, None)

# ===== XGBoost =====
print("⚡ XGBoost 학습 중...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    early_stopping_rounds=30, eval_metric="mae",
)
xgb_model.fit(X_train, y_train_log,
              eval_set=[(X_test, np.log1p(y_test))], verbose=False)
xgb_pred = np.clip(np.expm1(xgb_model.predict(X_test)), 0, None)

# ===== 비교 =====
def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "MAPE": mean_absolute_percentage_error(y, p) * 100,
        "R2": r2_score(y, p),
    }

rf_m = metrics(y_test, rf_pred)
xgb_m = metrics(y_test, xgb_pred)

print("\n" + "="*60)
print(f"{'지표':<12}{'Random Forest':<20}{'XGBoost':<20}{'승자':<10}")
print("="*60)
for k in ["MAE", "MAPE", "R2"]:
    rf_v, xgb_v = rf_m[k], xgb_m[k]
    # R²는 클수록 좋음, MAE/MAPE는 작을수록 좋음
    if k == "R2":
        winner = "RF" if rf_v > xgb_v else "XGB"
    else:
        winner = "RF" if rf_v < xgb_v else "XGB"
    print(f"{k:<12}{rf_v:<20,.3f}{xgb_v:<20,.3f}{winner:<10}")
print("="*60)

# R²로 최종 모델 선택
if rf_m["R2"] > xgb_m["R2"]:
    chosen, chosen_name, chosen_pred, chosen_m = rf, "RandomForest", rf_pred, rf_m
else:
    chosen, chosen_name, chosen_pred, chosen_m = xgb_model, "XGBoost", xgb_pred, xgb_m

print(f"\n🏆 최종 선택 모델: {chosen_name}")
print(f"   R² = {chosen_m['R2']:.3f}, MAE = {chosen_m['MAE']:,.0f}명, MAPE = {chosen_m['MAPE']:.1f}%")

# 최종 모델 저장 (기존 model.pkl 덮어쓰기)
joblib.dump(chosen, ROOT / "models" / "model.pkl")

# 메타데이터 업데이트 (모델 이름 추가)
meta = joblib.load(ROOT / "models" / "metadata.pkl")
meta["model_name"] = chosen_name
meta["test_mae"] = float(chosen_m["MAE"])
meta["test_mape"] = float(chosen_m["MAPE"])
meta["test_r2"] = float(chosen_m["R2"])
meta["comparison"] = {
    "random_forest": {k: float(v) for k, v in rf_m.items()},
    "xgboost": {k: float(v) for k, v in xgb_m.items()},
}
joblib.dump(meta, ROOT / "models" / "metadata.pkl")
print(f"\n✅ 저장 완료: models/model.pkl, models/metadata.pkl")
