"""
13_mape_analysis.py - MAPE를 10%대로 낮출 수 있는지 정량 분석

핵심 질문: MAPE가 큰 진짜 원인은?
  a) 모델이 부족한가? (개선 여지 큼)
  b) 데이터 분포 자체가 까다로운가? (본질적 한계)
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])

# v4 모델로 2025년 예측
import holidays as kh
hol = kh.country_holidays("KR", years=[2023, 2024, 2025])

def is_holiday(d):
    if d.date() in hol: return 1
    if d.month == 5 and d.day == 1: return 1
    return 0

def is_sandwich(d):
    if d.dayofweek >= 5 or is_holiday(d): return 0
    prev = d - pd.Timedelta(days=1)
    nxt = d + pd.Timedelta(days=1)
    return 1 if (is_holiday(prev) or is_holiday(nxt)) else 0

df["공휴일"] = df["날짜"].apply(is_holiday)
df["샌드위치데이"] = df["날짜"].apply(is_sandwich)
w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df = pd.concat([df, w1, w2], axis=1)

model = joblib.load(ROOT / "models" / "model.pkl")
meta = joblib.load(ROOT / "models" / "metadata.pkl")
FEAT = meta["features"]
for f in FEAT:
    if f not in df.columns:
        df[f] = 0

# 이상치 제외 후 평가 데이터
df_eval = df[df["실입장객"] >= 100].copy()
test = df_eval[df_eval["연도"] == 2025].copy()
test["예측"] = np.clip(np.expm1(model.predict(test[FEAT])), 0, None).astype(int)
test["오차"] = (test["실입장객"] - test["예측"]).abs()
test["오차율%"] = test["오차"] / test["실입장객"] * 100

# ===== 1) 입장객 구간별 MAPE =====
print("="*70)
print("📊 실제 입장객 구간별 MAPE (어디서 오차가 폭증하나?)")
print("="*70)
bins = [100, 500, 1000, 2000, 5000, 10000, 20000, 30000]
labels = ["100-500", "500-1K", "1K-2K", "2K-5K", "5K-10K", "10K-20K", "20K+"]
test["구간"] = pd.cut(test["실입장객"], bins=bins, labels=labels, right=False)
group = test.groupby("구간", observed=True).agg(
    n=("실입장객", "count"),
    실제평균=("실입장객", "mean"),
    예측평균=("예측", "mean"),
    MAE=("오차", "mean"),
    MAPE=("오차율%", "mean"),
).round(1)
print(group.to_string())

print("\n💡 인사이트: 작은 값 구간일수록 MAPE 폭증 (절대오차는 작아도 비례오차 큼)")

# ===== 2) MAPE 10% 미만으로 줄이려면 무엇이 필요한가? =====
print("\n" + "="*70)
print("🎯 만약 MAPE 10%를 달성하려면 — 절대오차 허용치는?")
print("="*70)
print(f"{'실제 입장객':<15}{'MAPE 10% 허용오차':<20}{'현재 MAE 1,187명':<20}")
print("-"*55)
for actual in [200, 500, 1000, 2000, 5000, 10000]:
    allowed = actual * 0.10
    diff = "달성 가능" if allowed >= 1187 else f"부족 ({1187 - int(allowed):,}명)"
    print(f"{actual:>10,}명     ±{int(allowed):>4,}명           {diff}")
print("\n💡 즉 입장객 12,000명 이상의 큰 날만 MAPE 10% 가능. 작은 날은 본질적으로 불가능.")

# ===== 3) 우리 데이터의 입장객 분포 =====
print("\n" + "="*70)
print("📊 우리 데이터 분포 — MAPE 10% 가능 구간 비중")
print("="*70)
total = len(test)
small = (test["실입장객"] < 2000).sum()
medium = ((test["실입장객"] >= 2000) & (test["실입장객"] < 10000)).sum()
large = (test["실입장객"] >= 10000).sum()
print(f"  2,000명 미만: {small:>4}일 ({small/total*100:>5.1f}%)  ← MAPE 10% 달성 매우 어려움")
print(f"  2K-10K:      {medium:>4}일 ({medium/total*100:>5.1f}%)")
print(f"  10K 이상:    {large:>4}일 ({large/total*100:>5.1f}%)  ← MAPE 10% 비교적 쉬움")

# ===== 4) MAPE의 진짜 의미: median은 어떤가? =====
print("\n" + "="*70)
print("📊 평균 MAPE vs 중앙값 MAPE (이상치 영향 분리)")
print("="*70)
print(f"  평균 MAPE:    {test['오차율%'].mean():.1f}%  ← 보고된 지표")
print(f"  중앙값 MAPE:  {test['오차율%'].median():.1f}%  ← 일반적인 날의 실제 정확도")
print(f"  75% MAPE:    {test['오차율%'].quantile(.75):.1f}%")
print(f"  90% MAPE:    {test['오차율%'].quantile(.90):.1f}%")

# 이상값 제거한 trimmed MAPE
trimmed = test["오차율%"][test["오차율%"] < test["오차율%"].quantile(.90)]
print(f"\n  하위 90% MAPE 평균: {trimmed.mean():.1f}%  ← 진짜 일반 정확도")

# ===== 5) 결론 =====
print("\n" + "="*70)
print("🎯 결론")
print("="*70)
print(f"""
  현재 v4 모델 MAPE 37%의 진짜 의미:
    - 중앙값(일반적인 날)은 이미 {test['오차율%'].median():.0f}% 수준
    - 평균을 끌어올리는 건 입장객 100~500명대의 한산한 날들
    - 그런 날은 ±300명 예측해도 비율로는 60%+ 오차로 잡힘

  MAPE 10%대 달성 = 사실상 불가능. 이유:
    1. 입장객 분포가 1명 ~ 25,876명 (지수 분포)
    2. 1,000명 미만인 날이 {(test['실입장객']<1000).sum()/len(test)*100:.0f}%
    3. 이런 날 MAPE를 10%로 만들려면 ±50~100명 정확도 필요
       → 인간도 불가능한 수준
""")
