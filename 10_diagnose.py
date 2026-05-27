"""
10_diagnose.py - 이상치/결측치 진단 보고서

v2 데이터를 깊이 들여다보고 정확도 발목 잡는 행을 찾아냅니다.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])

print("="*70)
print("📋 1. 결측치 점검")
print("="*70)
missing = df.isna().sum()
if missing.sum() == 0:
    print("✅ 결측치 0건 (모든 컬럼)")
else:
    print(missing[missing > 0])

print("\n" + "="*70)
print("📊 2. 실입장객 분포 (타겟)")
print("="*70)
print(df["실입장객"].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_string())

print("\n  하위 10개 (가장 적은 날):")
print(df.nsmallest(10, "실입장객")[
    ["날짜", "요일", "공휴일", "운영시간_시간", "일강수량", "날씨", "실입장객"]
].to_string(index=False))

print("\n  상위 5개 (가장 많은 날):")
print(df.nlargest(5, "실입장객")[
    ["날짜", "요일", "공휴일", "운영시간_시간", "일강수량", "날씨", "실입장객"]
].to_string(index=False))

print("\n" + "="*70)
print("📊 3. 운영시간 분포 (이 변수가 모델 1위 중요도)")
print("="*70)
print(df["운영시간_시간"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_string())
print(f"\n  4시간 이하 운영일 수: {(df['운영시간_시간'] <= 4).sum()}일")
print(f"  6시간 이하 운영일 수: {(df['운영시간_시간'] <= 6).sum()}일")
print(f"  13시간 초과 운영일 수: {(df['운영시간_시간'] > 13).sum()}일")
print(f"\n  4시간 이하 운영일 샘플:")
short = df[df["운영시간_시간"] <= 4].head(8)
print(short[["날짜", "요일", "운영시간_시간", "실입장객", "날씨"]].to_string(index=False))

print("\n" + "="*70)
print("📊 4. 강수량 분포")
print("="*70)
print(df["일강수량"].describe(percentiles=[.5, .9, .95, .99]).to_string())
print(f"\n  50mm 이상 호우일: {(df['일강수량'] >= 50).sum()}일")
print(df[df["일강수량"] >= 50][["날짜", "일강수량", "실입장객", "날씨"]].to_string(index=False))

print("\n" + "="*70)
print("🚨 5. 이상치 후보 (IQR 1.5배 기준)")
print("="*70)
# 입장객 IQR 이상치 (위쪽만 — 폭발일)
q1, q3 = df["실입장객"].quantile([.25, .75])
iqr = q3 - q1
high_thr = q3 + 1.5 * iqr
print(f"  IQR 기준 상한: {high_thr:.0f}명")
print(f"  상한 초과 (이상치 후보): {(df['실입장객'] > high_thr).sum()}일")
print(f"  → 그러나 어린이날/공휴일 등 정상 이벤트일 가능성 큼\n")

# 진짜 의심스러운 케이스: 비주말+비공휴일+비방학인데 폭발한 날
suspicious = df[
    (df["주말여부"] == 0) & (df["공휴일"] == 0) & (df["방학"] == 0)
    & (df["실입장객"] > high_thr)
]
print(f"  ⚠️ 평일+비공휴일+비방학인데 폭발한 날: {len(suspicious)}일")
if len(suspicious) > 0:
    print(suspicious[["날짜", "요일", "실입장객", "운영시간_시간", "날씨"]].head(10).to_string(index=False))

# 짧은 운영시간(< 6h)인데 입장객 많은 경우 (데이터 오류 의심)
weird = df[(df["운영시간_시간"] < 6) & (df["실입장객"] > 5000)]
print(f"\n  ⚠️ 운영시간 짧은데(<6h) 입장객 많은 날(>5000): {len(weird)}일")
if len(weird) > 0:
    print(weird[["날짜", "운영시간_시간", "실입장객", "날씨"]].to_string(index=False))

# 입장객 극소값 (운영했는데 거의 0인 경우)
tiny = df[df["실입장객"] < 50]
print(f"\n  ⚠️ 입장객 50명 미만 (휴장 직전/오기 의심): {len(tiny)}일")
if len(tiny) > 0:
    print(tiny[["날짜", "요일", "운영시간_시간", "일강수량", "날씨", "실입장객"]].to_string(index=False))

print("\n" + "="*70)
print("📊 6. 모델이 가장 크게 틀린 날 Top 10 (절대 오차 기준)")
print("="*70)
# v2 모델로 전체 데이터 재예측 후 오차 큰 날 확인
w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df_x = pd.concat([df, w1, w2], axis=1)

model = joblib.load(ROOT / "models" / "model.pkl")
meta = joblib.load(ROOT / "models" / "metadata.pkl")
FEATURES = meta["features"]
# 모든 feature 컬럼이 있는지 확인
for f in FEATURES:
    if f not in df_x.columns:
        df_x[f] = 0

pred = np.clip(np.expm1(model.predict(df_x[FEATURES])), 0, None)
df["예측"] = pred.astype(int)
df["오차"] = (df["실입장객"] - df["예측"]).abs()
df["오차율%"] = (df["오차"] / df["실입장객"].clip(1) * 100).round(1)

top_errors = df.nlargest(10, "오차")[
    ["날짜", "요일", "공휴일", "운영시간_시간", "날씨", "일강수량", "실입장객", "예측", "오차", "오차율%"]
]
print(top_errors.to_string(index=False))
