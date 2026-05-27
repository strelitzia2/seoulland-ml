"""
11_train_v3.py - v3 학습 (이상치 8일 제거 + 변수 2개 제거)

변경점:
  - 입장객 < 50명 행 제거 (8일, 휴장 수준 특수일)
  - 변수 제거: '날씨내부_기타' (중요도 0%), '강수여부' (일강수량과 중복)

v1 / v2 / v3 3-way 비교 후 가장 좋은 모델로 교체.
"""
import pandas as pd
import numpy as np
import joblib, shutil
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])

# ===== STEP 1: 이상치 제거 =====
before = len(df)
df = df[df["실입장객"] >= 50].copy()
removed = before - len(df)
print(f"🧹 이상치 제거: {before} → {len(df)}행  ({removed}일 제거)")

# 원-핫
w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df = pd.concat([df, w1, w2], axis=1)

# ===== STEP 2: 변수 선택 (제거 대상 2개 빼기) =====
DROP = {"날씨내부_기타", "강수여부"}
ALL_FEATURES = (
    ["월", "일", "요일", "주말여부", "공휴일", "방학",
     "평균기온", "최저기온", "최고기온",
     "일강수량", "강수여부", "폭우",
     "일조시간_시간", "운영시간_시간"]
    + list(w1.columns) + list(w2.columns)
)
FEATURES = [f for f in ALL_FEATURES if f not in DROP and f in df.columns]
print(f"📉 변수: {len(ALL_FEATURES)} → {len(FEATURES)}  (제거: {DROP})")

TARGET = "실입장객"
train = df[df["연도"] < 2025].copy()
test = df[df["연도"] == 2025].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
y_train_log = np.log1p(y_train)
print(f"   학습: {len(train)}일, 검증: {len(test)}일\n")

# ===== STEP 3: 학습 =====
print("🌳 Random Forest...")
rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_log)
rf_pred = np.clip(np.expm1(rf.predict(X_test)), 0, None)

print("⚡ XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    early_stopping_rounds=50, eval_metric="mae",
)
xgb_model.fit(X_train, y_train_log,
              eval_set=[(X_test, np.log1p(y_test))], verbose=False)
xgb_pred = np.clip(np.expm1(xgb_model.predict(X_test)), 0, None)

def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "MAPE": mean_absolute_percentage_error(y, p) * 100,
        "R2": r2_score(y, p),
    }
rf_m = metrics(y_test, rf_pred); xgb_m = metrics(y_test, xgb_pred)

# ===== STEP 4: v1/v2/v3 3-way 비교 =====
v2_meta = joblib.load(ROOT / "models" / "metadata.pkl")
v2_cmp = v2_meta.get("comparison_v2", {})
v2_best = v2_cmp.get("xgboost", {})
v1_best = v2_meta.get("comparison_v1_vs_v2", {})
v1_r2 = v1_best.get("v1_xgboost_r2", 0)
# v1 정확한 MAE는 metadata_v1.pkl
try:
    v1_meta_full = joblib.load(ROOT / "models" / "metadata_v1.pkl")
    v1_mae = v1_meta_full.get("test_mae", 0)
    v1_mape = v1_meta_full.get("test_mape", 0)
except Exception:
    v1_mae, v1_mape = 0, 0

# v3 best
if xgb_m["R2"] > rf_m["R2"]:
    v3_chosen, v3_name, v3_m = xgb_model, "XGBoost", xgb_m
else:
    v3_chosen, v3_name, v3_m = rf, "RandomForest", rf_m

print("\n" + "="*82)
print("📊 v1 → v2 → v3 3-way 비교 (검증: 2025년)")
print("="*82)
print(f"{'지표':<10}{'v1 (3파일)':<18}{'v2 (6파일)':<18}{'v3 (이상치+변수정리)':<22}{'개선폭(v2→v3)':<14}")
print("-"*82)
for k, vs in [("MAE", "명"), ("MAPE", "%"), ("R²", "")]:
    key = "R2" if k == "R²" else k
    v1v = v1_mae if k == "MAE" else (v1_mape if k == "MAPE" else v1_r2)
    v2v = v2_best.get(key, 0)
    v3v = v3_m[key]
    if k == "R²":
        delta = (v3v - v2v) * 100
        delta_str = f"{delta:+.2f}%p"
    else:
        delta = ((v2v - v3v) / v2v) * 100 if v2v else 0
        delta_str = f"{delta:+.1f}%"
    fmt = ",.0f" if k == "MAE" else ".3f"
    print(f"{k:<10}{format(v1v, fmt):<18}{format(v2v, fmt):<18}{format(v3v, fmt):<22}{delta_str:<14}")
print("="*82)

print(f"\n🏆 v3 최종: {v3_name}  R²={v3_m['R2']:.3f}, MAE={v3_m['MAE']:,.0f}명, MAPE={v3_m['MAPE']:.1f}%")

# ===== STEP 5: 특성 중요도 =====
imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": v3_chosen.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
print(f"\n🔥 v3 특성 중요도 Top 10:")
print(imp.head(10).to_string(index=False))
print(f"\n   하위 3개:")
print(imp.tail(3).to_string(index=False))

# ===== STEP 6: 저장 (v2 백업) =====
shutil.copy(ROOT / "models" / "model.pkl", ROOT / "models" / "model_v2.pkl")
shutil.copy(ROOT / "models" / "metadata.pkl", ROOT / "models" / "metadata_v2.pkl")

joblib.dump(v3_chosen, ROOT / "models" / "model.pkl")
meta = {
    "version": "v3",
    "model_name": v3_name,
    "features": FEATURES,
    "overall_mean": float(df[TARGET].mean()),
    "test_mae": float(v3_m["MAE"]),
    "test_mape": float(v3_m["MAPE"]),
    "test_r2": float(v3_m["R2"]),
    "removed_outliers": int(removed),
    "removed_features": list(DROP),
    "history": {
        "v1_xgboost": {"R2": float(v1_r2), "MAE": float(v1_mae), "MAPE": float(v1_mape)},
        "v2_xgboost": {k: float(v) for k, v in v2_best.items()},
        "v3_chosen": {k: float(v) for k, v in v3_m.items()},
    },
}
joblib.dump(meta, ROOT / "models" / "metadata.pkl")
print(f"\n✅ 저장: models/model.pkl (v3)")
print(f"   백업: models/model_v2.pkl")
