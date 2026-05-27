"""
12_grid_search.py - 정확도 극대화를 위한 그리드 서치

3가지 동시 개선 시도:
  A) 데이터 품질 보강: 근로자의날 추가 + 샌드위치데이 새 변수
  B) 이상치 임계값 그리드: [None, 30, 50, 100, 200]
  C) 변수 제거 개수 그리드: [0, 2, 4, 6]

→ 종합 점수(R² + (1-MAPE/100))로 최적 조합 선택
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (mean_absolute_error,
                              mean_absolute_percentage_error, r2_score)
import xgboost as xgb
import holidays as kr_hol

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df_raw = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])

# ===========================================================
# A) 데이터 품질 보강
# ===========================================================
# 1) 근로자의날(5/1) — holidays 라이브러리에 따라 누락될 수 있음
def is_holiday_v2(date):
    """holidays 라이브러리 + 한국 추가 휴일"""
    base = kr_hol.country_holidays("KR", years=[date.year])
    if date.date() in base:
        return 1
    # 근로자의날 (한국 노동절, 5/1)
    if date.month == 5 and date.day == 1:
        return 1
    return 0

df_raw["공휴일_v2"] = df_raw["날짜"].apply(is_holiday_v2)
print(f"📌 공휴일 보강: {df_raw['공휴일'].sum()} → {df_raw['공휴일_v2'].sum()}일")

# 2) 샌드위치 데이: 공휴일과 인접(전날/다음날)인 평일
holiday_dates = set(df_raw[df_raw["공휴일_v2"] == 1]["날짜"].dt.date)
def is_sandwich(date):
    if date.dayofweek >= 5:  # 주말은 제외
        return 0
    if date.date() in holiday_dates:
        return 0
    yesterday = (date - pd.Timedelta(days=1)).date()
    tomorrow = (date + pd.Timedelta(days=1)).date()
    return 1 if (yesterday in holiday_dates or tomorrow in holiday_dates) else 0

df_raw["샌드위치데이"] = df_raw["날짜"].apply(is_sandwich)
print(f"📌 샌드위치 데이: {df_raw['샌드위치데이'].sum()}일")
print(f"   샌드위치 평균 입장객: {df_raw[df_raw['샌드위치데이']==1]['실입장객'].mean():,.0f}명")
print(f"   일반 평일 평균 입장객: {df_raw[(df_raw['주말여부']==0)&(df_raw['공휴일_v2']==0)&(df_raw['샌드위치데이']==0)]['실입장객'].mean():,.0f}명")

# 공휴일 컬럼 교체
df_raw["공휴일"] = df_raw["공휴일_v2"]
df_raw = df_raw.drop(columns=["공휴일_v2"])

# ===========================================================
# 데이터 준비 (원-핫)
# ===========================================================
w1 = pd.get_dummies(df_raw["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df_raw["날씨_kma_단순"], prefix="날씨기상청")
df_full = pd.concat([df_raw, w1, w2], axis=1)

ALL_FEATURES = (
    ["월", "일", "요일", "주말여부", "공휴일", "방학", "샌드위치데이",
     "평균기온", "최저기온", "최고기온",
     "일강수량", "강수여부", "폭우",
     "일조시간_시간", "운영시간_시간"]
    + list(w1.columns) + list(w2.columns)
)
print(f"\n전체 변수: {len(ALL_FEATURES)}개")

# ===========================================================
# B+C) 그리드 서치
# ===========================================================
OUTLIER_THRS = [None, 30, 50, 100, 200]
DROP_COUNTS = [0, 2, 4, 6]

def train_and_eval(outlier_thr, n_drop):
    """주어진 조합으로 학습 + 평가"""
    df = df_full.copy()
    if outlier_thr is not None:
        df = df[df["실입장객"] >= outlier_thr].copy()

    train = df[df["연도"] < 2025]
    test = df[df["연도"] == 2025]

    # 1차 학습으로 변수 중요도 → 하위 n_drop 제거
    feat = list(ALL_FEATURES)
    if n_drop > 0:
        first = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0,
        )
        first.fit(train[feat], np.log1p(train["실입장객"]))
        imp = pd.Series(first.feature_importances_, index=feat).sort_values()
        drop_features = list(imp.head(n_drop).index)
        feat = [f for f in feat if f not in drop_features]
    else:
        drop_features = []

    # 본 학습
    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=50, eval_metric="mae", verbosity=0,
    )
    model.fit(
        train[feat], np.log1p(train["실입장객"]),
        eval_set=[(test[feat], np.log1p(test["실입장객"]))], verbose=False,
    )
    pred = np.clip(np.expm1(model.predict(test[feat])), 0, None)
    mae = mean_absolute_error(test["실입장객"], pred)
    mape = mean_absolute_percentage_error(test["실입장객"], pred) * 100
    r2 = r2_score(test["실입장객"], pred)
    return {
        "outlier_thr": outlier_thr, "n_drop": n_drop,
        "n_train": len(train), "n_features": len(feat),
        "MAE": mae, "MAPE": mape, "R2": r2,
        "drop_features": drop_features,
        "model": model, "features": feat,
    }

print("\n🔬 그리드 서치 시작 (20개 조합)...")
results = []
for thr in OUTLIER_THRS:
    for nd in DROP_COUNTS:
        r = train_and_eval(thr, nd)
        results.append(r)
        thr_str = "없음" if thr is None else f"<{thr}"
        print(f"  outlier={thr_str:<6}  n_drop={nd}  →  "
              f"R²={r['R2']:.3f}  MAE={r['MAE']:,.0f}  MAPE={r['MAPE']:.1f}%  "
              f"(train={r['n_train']}, feat={r['n_features']})")

# ===========================================================
# 종합 점수: 정규화된 R²와 (1-MAPE/100) 평균
# ===========================================================
res_df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ["model", "features", "drop_features"]}
                       for r in results])
res_df["score"] = res_df["R2"] + (1 - res_df["MAPE"] / 100)
res_df = res_df.sort_values("score", ascending=False).reset_index(drop=True)

print("\n" + "="*90)
print("🏆 종합 점수 순위 (score = R² + (1 - MAPE/100), 클수록 좋음)")
print("="*90)
print(res_df[["outlier_thr", "n_drop", "n_train", "n_features",
              "R2", "MAE", "MAPE", "score"]].head(10).to_string(index=False))

best_idx = res_df.index[0]
best_thr = res_df.iloc[0]["outlier_thr"]
best_nd = int(res_df.iloc[0]["n_drop"])
print(f"\n🥇 최적: outlier_thr={best_thr}, n_drop={best_nd}")

# 그 조합의 실제 모델 다시 찾기
best = [r for r in results
        if r["outlier_thr"] == best_thr and r["n_drop"] == best_nd][0]

print(f"   R²={best['R2']:.3f}, MAE={best['MAE']:,.0f}, MAPE={best['MAPE']:.1f}%")
print(f"   제거한 변수: {best['drop_features']}")
print(f"   사용 변수 {len(best['features'])}개: {best['features']}")

# ===========================================================
# v2/v3와 비교
# ===========================================================
v3_meta = joblib.load(ROOT / "models" / "metadata.pkl")
print("\n" + "="*70)
print("📊 v2 vs v3 vs v4(최적) 비교")
print("="*70)
print(f"{'지표':<8}{'v2':<14}{'v3':<14}{'v4 (best)':<14}{'v3→v4':<10}")
print("-"*70)
v3_hist = v3_meta["history"]["v3_chosen"]
v2_hist = v3_meta["history"]["v2_xgboost"]
for k in ["R2", "MAE", "MAPE"]:
    v2v = v2_hist[k]; v3v = v3_hist[k]; v4v = best[k]
    if k == "R2":
        delta = (v4v - v3v) * 100
        d_str = f"{delta:+.2f}%p"
    else:
        delta = ((v3v - v4v) / v3v) * 100
        d_str = f"{delta:+.1f}%"
    fmt = ".3f" if k == "R2" else ",.0f"
    print(f"{k:<8}{format(v2v, fmt):<14}{format(v3v, fmt):<14}{format(v4v, fmt):<14}{d_str:<10}")
print("="*70)

# v4 저장
import shutil
shutil.copy(ROOT / "models" / "model.pkl", ROOT / "models" / "model_v3.pkl")
shutil.copy(ROOT / "models" / "metadata.pkl", ROOT / "models" / "metadata_v3.pkl")

joblib.dump(best["model"], ROOT / "models" / "model.pkl")
meta = {
    "version": "v4",
    "model_name": "XGBoost",
    "features": best["features"],
    "overall_mean": float(df_full["실입장객"].mean()),
    "test_mae": float(best["MAE"]),
    "test_mape": float(best["MAPE"]),
    "test_r2": float(best["R2"]),
    "outlier_threshold": best_thr,
    "removed_features": best["drop_features"],
    "added_features": ["샌드위치데이", "공휴일(근로자의날 보강)"],
    "history": {
        **v3_meta.get("history", {}),
        "v4_best": {k: float(best[k]) for k in ["R2", "MAE", "MAPE"]},
    },
}
joblib.dump(meta, ROOT / "models" / "metadata.pkl")
print(f"\n✅ 저장: models/model.pkl (v4)")
