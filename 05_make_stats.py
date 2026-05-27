"""
05_make_stats.py - 통계 파일 생성 (stats.json)

발표 자료 및 API 응답에서 활용할 기초 통계.
"""
import pandas as pd
import json
from pathlib import Path

ROOT = Path.home() / "Desktop" / "seoulland-ml"
df = pd.read_csv(ROOT / "data" / "seoulland_clean.csv", parse_dates=["날짜"])

stats = {
    "data_period": {
        "start": str(df["날짜"].min().date()),
        "end": str(df["날짜"].max().date()),
        "total_days": int(len(df)),
    },
    "overall_mean": int(round(df["실입장객"].mean())),
    "overall_median": int(round(df["실입장객"].median())),
    "overall_max": int(df["실입장객"].max()),
    "overall_min": int(df["실입장객"].min()),
    "weekday_mean": int(round(df[df["주말여부"] == 0]["실입장객"].mean())),
    "weekend_mean": int(round(df[df["주말여부"] == 1]["실입장객"].mean())),
    "holiday_mean": int(round(df[df["공휴일"] == 1]["실입장객"].mean())),
    "vacation_mean": int(round(df[df["방학"] == 1]["실입장객"].mean())),
    # 요일별 평균 (0=월요일, 6=일요일)
    "by_weekday": {
        str(k): int(round(v)) for k, v in df.groupby("요일")["실입장객"].mean().items()
    },
    # 월별 평균 (1~12)
    "by_month": {
        str(k): int(round(v)) for k, v in df.groupby("월")["실입장객"].mean().items()
    },
    # 날씨별 평균
    "by_weather": {
        k: int(round(v)) for k, v in df.groupby("날씨")["실입장객"].mean().items()
    },
}

out_path = ROOT / "data" / "stats.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"✅ 저장: {out_path}\n")
print(json.dumps(stats, ensure_ascii=False, indent=2))
