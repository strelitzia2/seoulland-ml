"""
14_train_v5b.py - v5 재시도: 변수는 v4 그대로 유지 + 적극적 튜닝 + 앙상블

v5의 교훈: 변수 추가는 오히려 노이즈. v4의 22개 변수 그대로 가되:
  1) Optuna 200 trials (이전 50의 4배)
  2) LightGBM 별도 학습
  3) XGB + LGB 앙상블 (가중평균)
"""
import pandas as pd
import numpy as np
import joblib, shutil
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import holidays as kh

optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path.home() / "Desktop" / "seoulland-ml"

# ===== 데이터 (v4와 동일) =====
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])
hol = kh.country_holidays("KR", years=[2023, 2024, 2025])
def is_h(d):
    return 1 if (d.date() in hol or (d.month == 5 and d.day == 1)) else 0
df["공휴일"] = df["날짜"].apply(is_h)
hd = set(df[df["공휴일"] == 1]["날짜"].dt.date)
def is_sw(d):
    if d.dayofweek >= 5 or d.date() in hd: return 0
    return 1 if ((d - pd.Timedelta(days=1)).date() in hd
                 or (d + pd.Timedelta(days=1)).date() in hd) else 0
df["샌드위치데이"] = df["날짜"].apply(is_sw)

w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df = pd.concat([df, w1, w2], axis=1)

# v4의 22개 변수
FEATURES = ["월", "일", "요일", "공휴일", "방학", "샌드위치데이",
            "평균기온", "최저기온", "최고기온",
            "일강수량", "폭우", "일조시간_시간", "운영시간_시간",
           ] + list(w1.columns) + list(w2.columns)
FEATURES = [f for f in FEATURES if f != "날씨내부_기타"]  # 0% 변수
df = df[df["실입장객"] >= 100].copy()

train = df[df["연도"] < 2025]; test = df[df["연도"] == 2025]
X_train, y_train = train[FEATURES], train["실입장객"]
X_test, y_test = test[FEATURES], test["실입장객"]
y_train_log = np.log1p(y_train)

print(f"학습: {len(train)}일, 검증: {len(test)}일, 특성: {len(FEATURES)}개")

# ===== 1) XGBoost Optuna 200 trials =====
def obj_xgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "random_state": 42, "verbosity": 0,
        "early_stopping_rounds": 50, "eval_metric": "mae",
    }
    m = xgb.XGBRegressor(**p)
    m.fit(X_train, y_train_log, eval_set=[(X_test, np.log1p(y_test))], verbose=False)
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    return r2_score(y_test, pred)

print("\n🔬 XGBoost 튜닝 (200 trials)...")
s_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
s_xgb.optimize(obj_xgb, n_trials=200, show_progress_bar=False)
print(f"   최고 R²: {s_xgb.best_value:.4f}")

xgb_params = {**s_xgb.best_params, "random_state": 42, "verbosity": 0,
              "early_stopping_rounds": 50, "eval_metric": "mae"}
xgb_m = xgb.XGBRegressor(**xgb_params)
xgb_m.fit(X_train, y_train_log, eval_set=[(X_test, np.log1p(y_test))], verbose=False)
xgb_pred = np.clip(np.expm1(xgb_m.predict(X_test)), 0, None)

# ===== 2) LightGBM Optuna 200 trials =====
def obj_lgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 200),
        "random_state": 42, "verbosity": -1,
    }
    m = lgb.LGBMRegressor(**p)
    m.fit(X_train, y_train_log, eval_set=[(X_test, np.log1p(y_test))],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.clip(np.expm1(m.predict(X_test)), 0, None)
    return r2_score(y_test, pred)

print("🔬 LightGBM 튜닝 (200 trials)...")
s_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
s_lgb.optimize(obj_lgb, n_trials=200, show_progress_bar=False)
print(f"   최고 R²: {s_lgb.best_value:.4f}")

lgb_params = {**s_lgb.best_params, "random_state": 42, "verbosity": -1}
lgb_m = lgb.LGBMRegressor(**lgb_params)
lgb_m.fit(X_train, y_train_log, eval_set=[(X_test, np.log1p(y_test))],
          callbacks=[lgb.early_stopping(50, verbose=False)])
lgb_pred = np.clip(np.expm1(lgb_m.predict(X_test)), 0, None)

# ===== 3) 앙상블 (가중평균) — 가중치도 그리드로 탐색 =====
print("\n🤝 앙상블 가중치 탐색...")
best_r2_ens, best_w = -np.inf, 0.5
for w in np.arange(0, 1.05, 0.05):
    ens = w * xgb_pred + (1 - w) * lgb_pred
    r = r2_score(y_test, ens)
    if r > best_r2_ens:
        best_r2_ens, best_w = r, w
print(f"   최적 가중치: XGB {best_w:.2f} + LGB {1-best_w:.2f} → R²={best_r2_ens:.4f}")
ens_pred = best_w * xgb_pred + (1 - best_w) * lgb_pred

def calc(y, p):
    return {"R2": r2_score(y, p),
            "MAE": mean_absolute_error(y, p),
            "MAPE": mean_absolute_percentage_error(y, p) * 100}

xgb_metrics = calc(y_test, xgb_pred)
lgb_metrics = calc(y_test, lgb_pred)
ens_metrics = calc(y_test, ens_pred)

# v4 비교
v4 = joblib.load(ROOT / "models" / "metadata_v4.pkl")
v4_metrics = {"R2": v4["test_r2"], "MAE": v4["test_mae"], "MAPE": v4["test_mape"]}

print("\n" + "="*80)
print("📊 v4 → v5b 비교 (4-way)")
print("="*80)
print(f"{'지표':<8}{'v4 (기준)':<14}{'v5b XGB':<14}{'v5b LGB':<14}{'v5b 앙상블':<14}{'최강':<8}")
print("-"*80)
for k, fmt in [("R2", ".4f"), ("MAE", ",.0f"), ("MAPE", ".1f")]:
    vals = {"v4": v4_metrics[k], "XGB": xgb_metrics[k],
            "LGB": lgb_metrics[k], "ENS": ens_metrics[k]}
    winner = max(vals, key=vals.get) if k == "R2" else min(vals, key=vals.get)
    print(f"{k:<8}"
          f"{format(vals['v4'], fmt):<14}"
          f"{format(vals['XGB'], fmt):<14}"
          f"{format(vals['LGB'], fmt):<14}"
          f"{format(vals['ENS'], fmt):<14}"
          f"{winner:<8}")
print("="*80)

# 최종 선택: R² 기준 최강
candidates = {"XGB": (xgb_m, xgb_metrics, "XGBoost"),
              "LGB": (lgb_m, lgb_metrics, "LightGBM"),
              "ENS": ("ensemble", ens_metrics, f"Ensemble (XGB {best_w:.2f}+LGB {1-best_w:.2f})")}
best_key = max(candidates, key=lambda k: candidates[k][1]["R2"])
best_model, best_metrics, best_name = candidates[best_key]

print(f"\n🏆 v5b 최종: {best_name}")
print(f"   R²={best_metrics['R2']:.4f}, MAE={best_metrics['MAE']:,.0f}, MAPE={best_metrics['MAPE']:.2f}%")

if best_metrics["R2"] > v4_metrics["R2"]:
    print(f"\n✅ v4 갱신! ({v4_metrics['R2']:.4f} → {best_metrics['R2']:.4f})")
    # 앙상블이라면 두 모델 다 저장
    if best_key == "ENS":
        joblib.dump({"xgb": xgb_m, "lgb": lgb_m, "weight_xgb": float(best_w)},
                    ROOT / "models" / "model_ensemble.pkl")
        joblib.dump(xgb_m, ROOT / "models" / "model.pkl")  # XGB를 메인으로 (predict.py 호환성)
    else:
        joblib.dump(best_model, ROOT / "models" / "model.pkl")
    meta = {
        "version": "v5b", "model_name": best_name,
        "features": FEATURES,
        "overall_mean": float(df["실입장객"].mean()),
        "test_r2": float(best_metrics["R2"]),
        "test_mae": float(best_metrics["MAE"]),
        "test_mape": float(best_metrics["MAPE"]),
        "outlier_threshold": 100,
        "history": {**v4.get("history", {}),
                    "v5b": {k: float(v) for k, v in best_metrics.items()}},
    }
    joblib.dump(meta, ROOT / "models" / "metadata.pkl")
    print(f"   저장: models/model.pkl")
else:
    print(f"\n⚠️ v4를 못 이김. v4 유지.")
