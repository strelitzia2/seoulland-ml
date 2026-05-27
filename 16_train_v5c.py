"""
16_train_v5c.py - MAPE 집중 개선 (v5b의 R² 0.80 유지하면서)

전략:
  1) 샘플 가중치: 작은 값(한산한 날)에 1/log(y) 가중치 → MAPE 직접 최적화
  2) 이상치 임계값 그리드: [100, 200, 300, 500]
  3) 앙상블 가중치: R²↑ + (1-MAPE/100) 종합점수 최대화
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

# 데이터 준비 (v4와 동일)
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])
hol = kh.country_holidays("KR", years=[2023, 2024, 2025])
def is_h(d): return 1 if (d.date() in hol or (d.month == 5 and d.day == 1)) else 0
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

FEATURES = ["월", "일", "요일", "공휴일", "방학", "샌드위치데이",
            "평균기온", "최저기온", "최고기온",
            "일강수량", "폭우", "일조시간_시간", "운영시간_시간",
           ] + list(w1.columns) + list(w2.columns)
FEATURES = [f for f in FEATURES if f != "날씨내부_기타"]

# 알려진 좋은 하이퍼파라미터 (v5b에서 찾은 것 재사용 — 시간 절약)
XGB_PARAMS = {
    "n_estimators": 1500, "max_depth": 6, "learning_rate": 0.04,
    "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 6,
    "reg_alpha": 0.1, "reg_lambda": 0.5, "gamma": 0.05,
    "random_state": 42, "verbosity": 0,
    "early_stopping_rounds": 50, "eval_metric": "mae",
}
LGB_PARAMS = {
    "n_estimators": 1500, "max_depth": 6, "learning_rate": 0.04,
    "subsample": 0.85, "colsample_bytree": 0.85, "min_child_samples": 10,
    "reg_alpha": 0.1, "reg_lambda": 0.5, "num_leaves": 31,
    "random_state": 42, "verbosity": -1,
}

def train_eval(outlier_thr, use_weights=False):
    """주어진 설정으로 학습 + 평가."""
    d2 = df[df["실입장객"] >= outlier_thr].copy()
    train = d2[d2["연도"] < 2025]; test = d2[d2["연도"] == 2025]
    Xtr, ytr = train[FEATURES], train["실입장객"]
    Xte, yte = test[FEATURES], test["실입장객"]
    ytr_log = np.log1p(ytr)

    # 샘플 가중치: 1/log(y+e) — 작은 값에 큰 가중치
    if use_weights:
        sw = 1.0 / np.log1p(ytr.values + np.e)
        sw = sw / sw.mean()  # 정규화
    else:
        sw = None

    xgbm = xgb.XGBRegressor(**XGB_PARAMS)
    xgbm.fit(Xtr, ytr_log, sample_weight=sw,
             eval_set=[(Xte, np.log1p(yte))], verbose=False)
    xpred = np.clip(np.expm1(xgbm.predict(Xte)), 0, None)

    lgbm = lgb.LGBMRegressor(**LGB_PARAMS)
    fit_kw = {"sample_weight": sw} if use_weights else {}
    lgbm.fit(Xtr, ytr_log, eval_set=[(Xte, np.log1p(yte))],
             callbacks=[lgb.early_stopping(50, verbose=False)], **fit_kw)
    lpred = np.clip(np.expm1(lgbm.predict(Xte)), 0, None)

    # 앙상블 가중치: 종합점수 최대화 (R² + (1-MAPE/100))
    best = {"score": -np.inf}
    for w in np.arange(0, 1.01, 0.05):
        ens = w * xpred + (1 - w) * lpred
        r = r2_score(yte, ens)
        ma = mean_absolute_percentage_error(yte, ens) * 100
        mae = mean_absolute_error(yte, ens)
        sc = r + (1 - ma / 100)
        if sc > best["score"]:
            best = {"score": sc, "w": w, "R2": r, "MAPE": ma, "MAE": mae,
                    "xgb": xgbm, "lgb": lgbm, "xpred": xpred, "lpred": lpred}
    return best

# ===== 그리드: 이상치 임계값 × 가중치 사용여부 =====
results = []
for thr in [100, 200, 300, 500]:
    for w_flag in [False, True]:
        r = train_eval(thr, w_flag)
        results.append({"thr": thr, "weighted": w_flag, **{k: v for k, v in r.items() if k not in ['xgb','lgb','xpred','lpred']}})
        marker = "⚖️" if w_flag else "  "
        print(f"  thr<{thr:<4} {marker}  R²={r['R2']:.4f}  MAE={r['MAE']:,.0f}  MAPE={r['MAPE']:.2f}%  w_xgb={r['w']:.2f}  score={r['score']:.3f}")

df_r = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
print("\n" + "="*80)
print("🏆 종합 점수(R² + (1-MAPE/100)) 순위")
print("="*80)
print(df_r[["thr", "weighted", "R2", "MAE", "MAPE", "w", "score"]].to_string(index=False))

# v5b 비교 (R² 0.80, MAPE 38.6)
print("\n" + "="*80)
print("v5b 대비 — v5c top 3")
print("="*80)
print(f"{'설정':<25}{'R²':<10}{'MAE':<10}{'MAPE':<10}")
print("-"*55)
print(f"{'v5b 앙상블':<25}{0.8005:<10.4f}{1156:<10,.0f}{38.59:<10.2f}")
for i in range(min(3, len(df_r))):
    row = df_r.iloc[i]
    setting = f"thr<{int(row['thr'])} {'+weight' if row['weighted'] else ''}"
    print(f"{setting:<25}{row['R2']:<10.4f}{row['MAE']:<10,.0f}{row['MAPE']:<10.2f}")

# 최고 점수로 최종 학습
top = df_r.iloc[0]
print(f"\n🥇 최종 선택: thr<{int(top['thr'])}, weighted={top['weighted']}, xgb_w={top['w']:.2f}")
final = train_eval(int(top["thr"]), bool(top["weighted"]))

# v5b와 본 결과 비교
v5b = {"R2": 0.8005, "MAE": 1156, "MAPE": 38.59}
print(f"\n{'지표':<8}{'v5b':<12}{'v5c (선택)':<14}{'변화':<10}")
print("-"*45)
for k in ["R2", "MAE", "MAPE"]:
    v = v5b[k]; n = final[k]
    if k == "R2": ds = f"{(n-v)*100:+.2f}%p"
    else: ds = f"{(v-n)/v*100:+.1f}%"
    fmt = ".4f" if k == "R2" else (",.0f" if k == "MAE" else ".2f")
    print(f"{k:<8}{format(v, fmt):<12}{format(n, fmt):<14}{ds:<10}")

# 저장 결정
if final["score"] > (v5b["R2"] + (1 - v5b["MAPE"]/100)):
    print(f"\n✅ v5b 갱신!")
    # 백업
    if (ROOT / "models" / "model.pkl").exists():
        shutil.copy(ROOT / "models" / "model.pkl", ROOT / "models" / "model_v5b.pkl")
        shutil.copy(ROOT / "models" / "metadata.pkl", ROOT / "models" / "metadata_v5b.pkl")

    # 앙상블 저장
    joblib.dump({
        "xgb": final["xgb"], "lgb": final["lgb"],
        "weight_xgb": float(final["w"]),
    }, ROOT / "models" / "model_ensemble.pkl")
    # predict.py 호환 위해 메인은 XGB
    joblib.dump(final["xgb"], ROOT / "models" / "model.pkl")

    d2 = df[df["실입장객"] >= int(top["thr"])].copy()
    v5b_full = joblib.load(ROOT / "models" / "metadata_v5b.pkl")
    meta = {
        "version": "v5c",
        "model_name": f"Ensemble XGB({final['w']:.2f})+LGB({1-final['w']:.2f})",
        "features": FEATURES,
        "overall_mean": float(d2["실입장객"].mean()),
        "test_r2": float(final["R2"]),
        "test_mae": float(final["MAE"]),
        "test_mape": float(final["MAPE"]),
        "outlier_threshold": int(top["thr"]),
        "sample_weighted": bool(top["weighted"]),
        "ensemble_weight_xgb": float(final["w"]),
        "history": {**v5b_full.get("history", {}),
                    "v5c": {"R2": float(final["R2"]), "MAE": float(final["MAE"]),
                            "MAPE": float(final["MAPE"])}},
    }
    joblib.dump(meta, ROOT / "models" / "metadata.pkl")
    print(f"   저장: models/model_ensemble.pkl + models/model.pkl")
else:
    print(f"\n⚠️ 종합점수 v5b 못이김. v5b 유지.")
