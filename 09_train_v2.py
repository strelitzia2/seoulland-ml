"""
09_train_v2.py - 기상청 데이터 포함 재학습 + v1과 비교
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])

# 두 종류 날씨 모두 원-핫
w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df = pd.concat([df, w1, w2], axis=1)

FEATURES = (
    ["월", "일", "요일", "주말여부", "공휴일", "방학",
     "평균기온", "최저기온", "최고기온",          # ⭐ 평균기온 추가
     "일강수량", "강수여부", "폭우",              # ⭐ 강수 변수 3개 추가
     "일조시간_시간", "운영시간_시간"]            # ⭐ 일조시간 추가
    + list(w1.columns) + list(w2.columns)
)
TARGET = "실입장객"

train = df[df["연도"] < 2025].copy()
test = df[df["연도"] == 2025].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
y_train_log = np.log1p(y_train)

print(f"학습: {len(train)}일, 검증: {len(test)}일, 특성: {len(FEATURES)}개\n")

# ===== 학습 =====
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

def m(y, p): return {
    "MAE": mean_absolute_error(y, p),
    "MAPE": mean_absolute_percentage_error(y, p) * 100,
    "R2": r2_score(y, p),
}
rf_m = m(y_test, rf_pred); xgb_m = m(y_test, xgb_pred)

# ===== v1과 비교 =====
v1_meta = joblib.load(ROOT / "models" / "metadata.pkl")
v1 = v1_meta.get("comparison", {}).get("xgboost", {})

print("\n" + "="*78)
print("📊 v1 (3파일) vs v2 (6파일, 기상청 추가) 비교")
print("="*78)
print(f"{'지표':<8}{'v1 XGBoost':<18}{'v2 RF':<18}{'v2 XGBoost':<18}{'최강':<10}")
print("-"*78)
for k in ["MAE", "MAPE", "R2"]:
    v1v = v1.get(k, float('nan'))
    rfv = rf_m[k]
    xgbv = xgb_m[k]
    cands = {"v1 XGB": v1v, "v2 RF": rfv, "v2 XGB": xgbv}
    if k == "R2":
        winner = max(cands, key=cands.get)
    else:
        winner = min(cands, key=cands.get)
    print(f"{k:<8}{v1v:<18,.3f}{rfv:<18,.3f}{xgbv:<18,.3f}{winner:<10}")
print("="*78)

# 더 나은 v2 모델 선택
if xgb_m["R2"] > rf_m["R2"]:
    chosen, name, cm = xgb_model, "XGBoost", xgb_m
else:
    chosen, name, cm = rf, "RandomForest", rf_m

print(f"\n🏆 v2 최종: {name}  (R²={cm['R2']:.3f}, MAE={cm['MAE']:,.0f}, MAPE={cm['MAPE']:.1f}%)")

# v1 대비 개선폭
v1_r2 = v1.get("R2", 0)
v1_mae = v1.get("MAE", 0)
improvement_r2 = (cm["R2"] - v1_r2) * 100
improvement_mae = ((v1_mae - cm["MAE"]) / v1_mae) * 100 if v1_mae else 0
print(f"\n📈 v1 대비 개선:")
print(f"   R² : {v1_r2:.3f} → {cm['R2']:.3f}  ({improvement_r2:+.1f}%p)")
print(f"   MAE: {v1_mae:,.0f}명 → {cm['MAE']:,.0f}명  ({improvement_mae:+.1f}%)")

# 특성 중요도 Top 15
imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": chosen.feature_importances_,
}).sort_values("importance", ascending=False)
print(f"\n🔥 v2 특성 중요도 Top 15:")
print(imp.head(15).to_string(index=False))

# ===== 저장 (v1은 백업) =====
import shutil
if (ROOT / "models" / "model.pkl").exists():
    shutil.copy(ROOT / "models" / "model.pkl", ROOT / "models" / "model_v1.pkl")
    shutil.copy(ROOT / "models" / "metadata.pkl", ROOT / "models" / "metadata_v1.pkl")

joblib.dump(chosen, ROOT / "models" / "model.pkl")
meta = {
    "version": "v2",
    "model_name": name,
    "features": FEATURES,
    "overall_mean": float(df[TARGET].mean()),
    "test_mae": float(cm["MAE"]),
    "test_mape": float(cm["MAPE"]),
    "test_r2": float(cm["R2"]),
    "comparison_v2": {
        "random_forest": {k: float(v) for k, v in rf_m.items()},
        "xgboost": {k: float(v) for k, v in xgb_m.items()},
    },
    "comparison_v1_vs_v2": {
        "v1_xgboost_r2": float(v1_r2),
        "v2_chosen_r2": float(cm["R2"]),
        "r2_gain_pp": float(improvement_r2),
    },
}
joblib.dump(meta, ROOT / "models" / "metadata.pkl")
print(f"\n✅ 저장: models/model.pkl (v2)")
print(f"   백업: models/model_v1.pkl")
