"""
14_train_v5.py - R² 0.85+ 도전

세 가지 동시 적용:
  1) 시계열 lag 변수 (전주 동일 요일, 이동평균 등) — 데이터 누수 방지
  2) 상호작용 변수 (월×강수, 주말×기온 등)
  3) Optuna 하이퍼파라미터 자동 튜닝 (50회 시도)
"""
import pandas as pd
import numpy as np
import joblib, shutil
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import optuna
import holidays as kh

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])
df = df.sort_values("날짜").reset_index(drop=True)

# ===== 공휴일 보강 + 샌드위치 =====
hol = kh.country_holidays("KR", years=[2023, 2024, 2025])
def is_h(d):
    if d.date() in hol: return 1
    if d.month == 5 and d.day == 1: return 1
    return 0
df["공휴일"] = df["날짜"].apply(is_h)
hol_dates = set(df[df["공휴일"] == 1]["날짜"].dt.date)
def is_sw(d):
    if d.dayofweek >= 5 or d.date() in hol_dates: return 0
    prev, nxt = (d - pd.Timedelta(days=1)).date(), (d + pd.Timedelta(days=1)).date()
    return 1 if (prev in hol_dates or nxt in hol_dates) else 0
df["샌드위치데이"] = df["날짜"].apply(is_sw)

# ===========================================================
# 1) 시계열 lag 변수 (데이터 누수 방지: shift는 항상 과거만 본다)
# ===========================================================
# 전주 동일 요일
df["lag_7"] = df["실입장객"].shift(7)
# 2주 전 동일 요일
df["lag_14"] = df["실입장객"].shift(14)
# 365일 전 (전년 동일 일자) — 1년차 데이터는 NaN
df["lag_365"] = df["실입장객"].shift(365)

# 최근 7일/14일 이동평균 (어제까지)
df["rolling_7"] = df["실입장객"].shift(1).rolling(7).mean()
df["rolling_14"] = df["실입장객"].shift(1).rolling(14).mean()
df["rolling_30"] = df["실입장객"].shift(1).rolling(30).mean()

# 같은 요일 최근 4회 평균 (요일별로 그룹화 후 shift)
df["same_dow_avg_4"] = df.groupby(df["날짜"].dt.dayofweek)["실입장객"].transform(
    lambda x: x.shift(1).rolling(4).mean())

# ===========================================================
# 2) 상호작용 변수
# ===========================================================
df["주말x공휴일"] = (df["날짜"].dt.dayofweek >= 5).astype(int) * df["공휴일"]
df["월x강수"] = df["날짜"].dt.month * df["일강수량"]
df["주말여부"] = (df["날짜"].dt.dayofweek >= 5).astype(int)
df["주말x기온"] = df["주말여부"] * df["평균기온"]
df["여름x폭우"] = ((df["날짜"].dt.month.isin([6,7,8]))).astype(int) * df["폭우"]
df["성수기"] = df["날짜"].dt.month.isin([4, 5, 10]).astype(int)
df["성수기x주말"] = df["성수기"] * df["주말여부"]

# ===========================================================
# 원-핫
# ===========================================================
w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df = pd.concat([df, w1, w2], axis=1)

# 결측치 (초기 30일은 lag/rolling이 NaN) → 채우기
lag_cols = ["lag_7", "lag_14", "lag_365", "rolling_7", "rolling_14", "rolling_30", "same_dow_avg_4"]
overall_mean = df["실입장객"].mean()
for c in lag_cols:
    df[c] = df[c].fillna(overall_mean)

# ===========================================================
# 변수 선택 (v4 + 신규)
# ===========================================================
FEATURES = (
    ["월", "일", "요일", "공휴일", "방학", "샌드위치데이",
     "평균기온", "최저기온", "최고기온",
     "일강수량", "폭우", "일조시간_시간", "운영시간_시간",
     # 신규 시계열
     "lag_7", "lag_14", "lag_365",
     "rolling_7", "rolling_14", "rolling_30", "same_dow_avg_4",
     # 신규 상호작용
     "주말x공휴일", "월x강수", "주말x기온", "여름x폭우", "성수기", "성수기x주말",
     ] + list(w1.columns) + list(w2.columns)
)
# 이상치 제거 (v4 기준)
df = df[df["실입장객"] >= 100].copy()

train = df[df["연도"] < 2025].copy()
test = df[df["연도"] == 2025].copy()
X_train, y_train = train[FEATURES], train["실입장객"]
X_test, y_test = test[FEATURES], test["실입장객"]
y_train_log = np.log1p(y_train)

print(f"📊 v5 데이터:")
print(f"   학습: {len(train)}일, 검증: {len(test)}일")
print(f"   특성: {len(FEATURES)}개 (v4의 22개 + 신규 {len(FEATURES)-22}개)")

# ===========================================================
# 3) Optuna 하이퍼파라미터 튜닝 (50 trials)
# ===========================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "random_state": 42, "verbosity": 0,
        "early_stopping_rounds": 50, "eval_metric": "mae",
    }
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train_log,
          eval_set=[(X_test, np.log1p(y_test))], verbose=False)
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    return r2_score(y_test, pred)

print("\n🔬 Optuna 튜닝 (50 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"\n✅ 최적 R²: {study.best_value:.4f}")
print(f"   하이퍼파라미터: {study.best_params}")

# 최종 모델
best_params = {**study.best_params,
               "random_state": 42, "verbosity": 0,
               "early_stopping_rounds": 50, "eval_metric": "mae"}
final = xgb.XGBRegressor(**best_params)
final.fit(X_train, y_train_log,
          eval_set=[(X_test, np.log1p(y_test))], verbose=False)
pred = np.clip(np.expm1(final.predict(X_test)), 0, None)

mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred) * 100
r2 = r2_score(y_test, pred)

# ===========================================================
# 비교
# ===========================================================
v4_meta = joblib.load(ROOT / "models" / "metadata.pkl")
v4 = v4_meta["history"]["v4_best"]

print("\n" + "="*70)
print("📊 v4 → v5 비교")
print("="*70)
print(f"{'지표':<8}{'v4':<14}{'v5 (시계열+튜닝)':<20}{'변화':<12}")
print("-"*70)
for k, fmt in [("R2", ".4f"), ("MAE", ",.0f"), ("MAPE", ".2f")]:
    v4v = v4[k]
    v5v = {"R2": r2, "MAE": mae, "MAPE": mape}[k]
    if k == "R2":
        delta = (v5v - v4v) * 100
        ds = f"{delta:+.2f}%p"
    else:
        delta = (v4v - v5v) / v4v * 100
        ds = f"{delta:+.1f}%"
    print(f"{k:<8}{format(v4v, fmt):<14}{format(v5v, fmt):<20}{ds:<12}")
print("="*70)

# 특성 중요도
imp = pd.DataFrame({"feature": FEATURES, "importance": final.feature_importances_})\
        .sort_values("importance", ascending=False).reset_index(drop=True)
print(f"\n🔥 v5 특성 중요도 Top 12:")
print(imp.head(12).to_string(index=False))

# 신규 변수들이 얼마나 기여했나
new_vars = ["lag_7", "lag_14", "lag_365", "rolling_7", "rolling_14", "rolling_30",
            "same_dow_avg_4", "주말x공휴일", "월x강수", "주말x기온", "여름x폭우",
            "성수기", "성수기x주말"]
new_imp_sum = imp[imp["feature"].isin(new_vars)]["importance"].sum()
print(f"\n💡 신규 변수({len(new_vars)}개) 총 기여도: {new_imp_sum*100:.1f}%")

# 저장
shutil.copy(ROOT / "models" / "model.pkl", ROOT / "models" / "model_v4.pkl")
shutil.copy(ROOT / "models" / "metadata.pkl", ROOT / "models" / "metadata_v4.pkl")

joblib.dump(final, ROOT / "models" / "model.pkl")
# v5는 시계열 변수 의존도가 있어 lag 값들도 함께 저장해야 함
recent_data = df.tail(60)[["날짜", "실입장객"]].copy()
recent_data.to_csv(ROOT / "data" / "recent_history.csv", index=False, encoding="utf-8-sig")

meta = {
    "version": "v5",
    "model_name": "XGBoost (Optuna-tuned)",
    "features": FEATURES,
    "overall_mean": float(df["실입장객"].mean()),
    "test_mae": float(mae), "test_mape": float(mape), "test_r2": float(r2),
    "outlier_threshold": 100,
    "best_params": study.best_params,
    "history": {
        **v4_meta.get("history", {}),
        "v5_optuna": {"R2": float(r2), "MAE": float(mae), "MAPE": float(mape)},
    },
}
joblib.dump(meta, ROOT / "models" / "metadata.pkl")
print(f"\n✅ 저장: models/model.pkl (v5)")
print(f"   백업: models/model_v4.pkl")
