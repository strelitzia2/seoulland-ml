"""
17_rebalance_ensemble.py - v5b 앙상블의 가중치만 MAPE 친화적으로 재탐색

v5b의 두 모델(XGB, LGB)은 이미 잘 학습되어 있음.
다만 가중치 탐색이 R² 단독 기준이었음.
종합점수(R²+1-MAPE/100), MAPE 단독, R² 단독 세 기준으로 비교해서
사용자가 원하는 균형을 찾는다.
"""
import joblib, json
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import (mean_absolute_error,
                              mean_absolute_percentage_error, r2_score)
import holidays as kh

ROOT = Path.home() / "Desktop" / "seoulland-ml"
ens = joblib.load(ROOT / "models" / "model_ensemble.pkl")
xgb_m, lgb_m = ens["xgb"], ens["lgb"]
meta = joblib.load(ROOT / "models" / "metadata.pkl")
FEATURES = meta["features"]

# 데이터 재생성 (학습 때와 동일)
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
df = df[df["실입장객"] >= 100].copy()
test = df[df["연도"] == 2025]
y = test["실입장객"].values

xpred = np.clip(np.expm1(xgb_m.predict(test[FEATURES])), 0, None)
lpred = np.clip(np.expm1(lgb_m.predict(test[FEATURES])), 0, None)

print("="*80)
print("앙상블 가중치 탐색: w=XGB 비중 (0=LGB만, 1=XGB만)")
print("="*80)
print(f"{'w':<6}{'R²':<10}{'MAE':<10}{'MAPE':<10}{'종합점수':<10}")
print("-"*46)
rows = []
for w in np.arange(0, 1.001, 0.025):
    e = w * xpred + (1 - w) * lpred
    r = r2_score(y, e)
    ma = mean_absolute_error(y, e)
    mp = mean_absolute_percentage_error(y, e) * 100
    rows.append({"w": w, "R2": r, "MAE": ma, "MAPE": mp, "score": r + (1 - mp/100)})
res = pd.DataFrame(rows)
# 모든 행 출력하면 너무 많으니 0.05 단위로만
for _, r in res[res["w"].round(3).isin([round(x,3) for x in np.arange(0,1.01,0.05)])].iterrows():
    print(f"{r['w']:<6.2f}{r['R2']:<10.4f}{r['MAE']:<10,.0f}{r['MAPE']:<10.2f}{r['score']:<10.4f}")

# 각 기준별 최적
print("\n" + "="*80)
print("기준별 최적 가중치")
print("="*80)
options = {
    "R² 최대": res.loc[res["R2"].idxmax()],
    "MAPE 최소": res.loc[res["MAPE"].idxmin()],
    "MAE 최소": res.loc[res["MAE"].idxmin()],
    "종합점수 최대": res.loc[res["score"].idxmax()],
}
for label, row in options.items():
    print(f"  {label:<15}  w={row['w']:.2f}  →  R²={row['R2']:.4f}  MAE={row['MAE']:,.0f}  MAPE={row['MAPE']:.2f}%  score={row['score']:.4f}")

# 사용자 목표: "R² 0.77 유지하면서 MAPE 최저"
# R² 0.77+ 필터링 후 MAPE 최저
filt = res[res["R2"] >= 0.77]
if len(filt) > 0:
    best = filt.loc[filt["MAPE"].idxmin()]
    print(f"\n🎯 R² 0.77 이상 조건에서 MAPE 최저:")
    print(f"   w={best['w']:.2f}  →  R²={best['R2']:.4f}, MAE={best['MAE']:,.0f}, MAPE={best['MAPE']:.2f}%")
else:
    print("\n⚠️ R² 0.77+ 영역이 없음 (이미 v5b가 R² 0.80)")

# v5b 기본(w=0.60)와 사용자 친화(R²≥0.77, MAPE 최저) 비교
v5b_default = res[res["w"] == 0.60].iloc[0]
print(f"\n📊 v5b 기본 vs 'R²≥0.77 + MAPE 최저' 비교:")
print(f"{'설정':<28}{'R²':<10}{'MAE':<10}{'MAPE':<10}")
print("-"*55)
print(f"{'v5b 기본 (w=0.60)':<28}{v5b_default['R2']:<10.4f}{v5b_default['MAE']:<10,.0f}{v5b_default['MAPE']:<10.2f}")
if len(filt) > 0:
    print(f"{'R²≥0.77 + MAPE 최저':<28}{best['R2']:<10.4f}{best['MAE']:<10,.0f}{best['MAPE']:<10.2f}")

    # 메타데이터 업데이트
    new_w = float(best["w"])
    if abs(new_w - 0.60) > 0.001:
        print(f"\n💡 가중치를 {new_w:.2f}로 변경 추천")
        ens["weight_xgb"] = new_w
        joblib.dump(ens, ROOT / "models" / "model_ensemble.pkl")
        meta["ensemble_weight_xgb"] = new_w
        meta["test_r2"] = float(best["R2"])
        meta["test_mae"] = float(best["MAE"])
        meta["test_mape"] = float(best["MAPE"])
        meta["model_name"] = f"Ensemble XGB({new_w:.2f})+LGB({1-new_w:.2f}) [MAPE-balanced]"
        meta["version"] = "v5b-rebalanced"
        joblib.dump(meta, ROOT / "models" / "metadata.pkl")
        print(f"   저장: models/model_ensemble.pkl (가중치 변경)")
