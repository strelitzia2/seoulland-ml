"""
06_visualize.py - 발표 PPT용 시각화 4장 생성

저장 위치: ~/Desktop/seoulland-ml/figures/
1. 실제 vs 예측 산점도
2. 월별 평균 입장객 추이
3. 날씨별 평균 입장객 막대
4. 특징 중요도 차트
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import platform

ROOT = Path.home() / "Desktop" / "seoulland-ml"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# === 한글 폰트 설정 (macOS 기본 'AppleGothic') ===
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지

# === 데이터 로드 ===
df = pd.read_csv(ROOT / "data" / "seoulland_clean_v2.csv", parse_dates=["날짜"])
w1 = pd.get_dummies(df["날씨"], prefix="날씨내부")
w2 = pd.get_dummies(df["날씨_kma_단순"], prefix="날씨기상청")
df_x = pd.concat([df, w1, w2], axis=1)

meta = joblib.load(ROOT / "models" / "metadata.pkl")
FEATURES = meta["features"]

# 앙상블 모델 폴백
ens_path = ROOT / "models" / "model_ensemble.pkl"
if ens_path.exists():
    ens = joblib.load(ens_path)
    xgb_m, lgb_m = ens["xgb"], ens["lgb"]
    w = ens.get("weight_xgb", 0.5)
    def predict_fn(X):
        return w * np.expm1(xgb_m.predict(X)) + (1 - w) * np.expm1(lgb_m.predict(X))
    model = xgb_m  # 특성 중요도용 (XGB)
else:
    model = joblib.load(ROOT / "models" / "model.pkl")
    def predict_fn(X):
        return np.expm1(model.predict(X))

# 데이터에 v5 신규 변수 추가 (샌드위치데이, 공휴일 보강)
import holidays as kh
hol = kh.country_holidays("KR", years=[2023, 2024, 2025])
def is_h(d): return 1 if (d.date() in hol or (d.month == 5 and d.day == 1)) else 0
df["공휴일"] = df["날짜"].apply(is_h)
hd = set(df[df["공휴일"] == 1]["날짜"].dt.date)
def is_sw(d):
    if d.dayofweek >= 5 or d.date() in hd: return 0
    return 1 if ((d - pd.Timedelta(days=1)).date() in hd
                 or (d + pd.Timedelta(days=1)).date() in hd) else 0
df["샌드위치데이"] = df["날짜"].apply(is_sw)
df_x["공휴일"] = df["공휴일"]
df_x["샌드위치데이"] = df["샌드위치데이"]
# 누락 컬럼 0으로
for f in FEATURES:
    if f not in df_x.columns:
        df_x[f] = 0

# === 2025년 예측 (검증 데이터) ===
df_x = df_x[df_x["실입장객"] >= 100]  # v5b 이상치 기준
test = df_x[df_x["연도"] == 2025].copy()
pred = np.clip(predict_fn(test[FEATURES]), 0, None)
actual = test["실입장객"].values

# ───────────────────────────────────────────────────────
# 1. 실제 vs 예측 산점도
# ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))
plt.scatter(actual, pred, alpha=0.5, s=30, c="steelblue")
m = max(actual.max(), pred.max())
plt.plot([0, m], [0, m], "r--", linewidth=2, label="완벽한 예측 (y=x)")
plt.xlabel("실제 입장객 수 (명)", fontsize=12)
plt.ylabel("예측 입장객 수 (명)", fontsize=12)
plt.title(f"실제 vs 예측 (2025년 검증) — R² = {meta['test_r2']:.3f}", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "1_actual_vs_predicted.png", dpi=150)
plt.close()
print("✅ 1_actual_vs_predicted.png")

# ───────────────────────────────────────────────────────
# 2. 월별 평균 입장객 추이
# ───────────────────────────────────────────────────────
monthly = df.groupby("월")["실입장객"].mean()
plt.figure(figsize=(10, 6))
bars = plt.bar(monthly.index, monthly.values,
               color=["#3498db" if v < monthly.mean() else "#e74c3c"
                      for v in monthly.values])
plt.axhline(monthly.mean(), color="gray", linestyle="--",
            label=f"연평균 {monthly.mean():.0f}명")
plt.xlabel("월", fontsize=12)
plt.ylabel("평균 입장객 수 (명)", fontsize=12)
plt.title("월별 평균 입장객 추이 (3년치 평균)", fontsize=14)
plt.xticks(range(1, 13))
plt.legend(fontsize=11)
plt.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, monthly.values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 100, f"{int(v):,}",
             ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "2_monthly_trend.png", dpi=150)
plt.close()
print("✅ 2_monthly_trend.png")

# ───────────────────────────────────────────────────────
# 3. 날씨별 평균 입장객
# ───────────────────────────────────────────────────────
weather_mean = df.groupby("날씨")["실입장객"].mean().sort_values(ascending=False)
plt.figure(figsize=(9, 6))
colors = ["#f1c40f", "#95a5a6", "#3498db", "#9b59b6", "#7f8c8d"][:len(weather_mean)]
bars = plt.bar(weather_mean.index, weather_mean.values, color=colors)
plt.xlabel("날씨 상태", fontsize=12)
plt.ylabel("평균 입장객 수 (명)", fontsize=12)
plt.title("날씨별 평균 입장객 (3년치)", fontsize=14)
plt.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, weather_mean.values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 100, f"{int(v):,}",
             ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "3_by_weather.png", dpi=150)
plt.close()
print("✅ 3_by_weather.png")

# ───────────────────────────────────────────────────────
# 4. 특징 중요도 (XGBoost)
# ───────────────────────────────────────────────────────
imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=True).tail(12)

plt.figure(figsize=(10, 7))
plt.barh(imp["feature"], imp["importance"], color="#2ecc71")
plt.xlabel("중요도 (XGBoost feature importance)", fontsize=12)
plt.title("입장객 예측에 영향을 미치는 주요 특성 Top 12", fontsize=14)
plt.grid(axis="x", alpha=0.3)
for i, (f, v) in enumerate(zip(imp["feature"], imp["importance"])):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "4_feature_importance.png", dpi=150)
plt.close()
print("✅ 4_feature_importance.png")

print(f"\n📁 저장 위치: {FIG_DIR}")
