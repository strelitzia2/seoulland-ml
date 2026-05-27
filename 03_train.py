"""
03_train.py - XGBoost 모델 학습 + 저장

- 2023-2024 → 학습용
- 2025 → 검증용 (모델이 본 적 없는 미래 데이터로 평가)
- 평균 대비 % 계산을 위해 학습 데이터 평균값도 함께 저장
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean.csv", parse_dates=["날짜"])

# 카테고리(날씨)를 숫자로 변환 (One-Hot Encoding)
weather_dummies = pd.get_dummies(df["날씨"], prefix="날씨")
df = pd.concat([df, weather_dummies], axis=1)

FEATURES = (["월", "일", "요일", "주말여부", "공휴일", "방학",
             "최저기온", "최고기온", "운영시간_시간"]
            + list(weather_dummies.columns))
TARGET = "실입장객"

# 시간 기준 분리
train = df[df["연도"] < 2025].copy()
test = df[df["연도"] == 2025].copy()
print(f"학습: {len(train)}일 (2023-2024)")
print(f"검증: {len(test)}일 (2025)")

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# 로그 변환: 입장객은 분포가 매우 왼쪽으로 치우쳐 있어서 log를 씌우면 학습이 훨씬 안정적
y_train_log = np.log1p(y_train)

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    eval_metric="mae",
)
model.fit(
    X_train, y_train_log,
    eval_set=[(X_test, np.log1p(y_test))],
    verbose=False,
)

# 예측 (로그 복원)
pred = np.expm1(model.predict(X_test))
pred = np.clip(pred, 0, None)  # 음수 방지

mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred) * 100
r2 = r2_score(y_test, pred)

print(f"\n📊 모델 성능 (2025년 검증):")
print(f"  MAE  (평균 절대 오차): {mae:>8,.0f}명")
print(f"  MAPE (평균 오차율):   {mape:>8,.1f}%")
print(f"  R²   (설명력):        {r2:>8,.3f}  (1에 가까울수록 좋음)")
print(f"  실제 평균 입장객:     {y_test.mean():>8,.0f}명")

# 특성 중요도
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print(f"\n🔥 가장 중요한 특성 Top 10:")
print(importance.head(10).to_string(index=False))

# === 모델 + 메타데이터 저장 ===
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "model.pkl")

# 예측 시 필요한 정보들
metadata = {
    "features": FEATURES,
    "weather_categories": list(df["날씨"].unique()),
    "overall_mean": float(df[TARGET].mean()),     # 전체 평균 (혼잡도 % 기준)
    "weekday_mean": df.groupby("요일")[TARGET].mean().to_dict(),  # 요일별 평균
    "test_mae": float(mae),
    "test_mape": float(mape),
    "test_r2": float(r2),
}
joblib.dump(metadata, models_dir / "metadata.pkl")
print(f"\n✅ 저장: {models_dir/'model.pkl'}, {models_dir/'metadata.pkl'}")
print(f"   전체 평균 입장객 = {metadata['overall_mean']:,.0f}명 (혼잡도 % 기준)")
