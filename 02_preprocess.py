"""
02_preprocess.py - 3년치 데이터 통합 + 파생변수 생성

36개 시트(3년 × 12월)를 하나의 깨끗한 데이터프레임으로 만듭니다.
- 헤더 3줄 제거
- 일자를 datetime으로 변환
- 공휴일/방학 등 파생변수 추가
- 결과: data/seoulland_clean.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import holidays
import re

DATA_DIR = Path.home() / "Desktop" / "Datasets" / "1.일별입장객및기상조건"
OUT_DIR = Path.home() / "Desktop" / "seoulland-ml" / "data"
OUT_DIR.mkdir(exist_ok=True)

# 통합 컬럼명 (헤더 3줄을 우리가 알고 있는 의미로 매핑)
COLS = ["일자", "요일", "날씨", "최저기온", "최고기온",
        "개장시간", "폐장시간", "매표입장객_주간", "실입장객"]

all_rows = []

for xlsx in sorted(DATA_DIR.glob("*.xlsx")):
    # 파일명에서 연도 추출: "일별입장객및기상조건(23년)..." → 2023
    m = re.search(r"(\d{2})", xlsx.name)  # 파일명 첫 2자리 숫자 = 연도
    if not m:
        print(f"⚠️ 연도 추출 실패: {xlsx.name}")
        continue
    year = 2000 + int(m.group(1))
    print(f"📄 {xlsx.name} → {year}년")

    for sheet in pd.ExcelFile(xlsx).sheet_names:
        # 월 번호 추출 ("1월" → 1)
        month = int(re.search(r"(\d+)", sheet).group(1))
        # 헤더 3줄 건너뛰기 (skiprows=3), 컬럼명 직접 지정
        df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=3, header=None, names=COLS)
        df["연도"] = year
        df["월"] = month
        all_rows.append(df)

raw = pd.concat(all_rows, ignore_index=True)
print(f"\n✅ 통합 행 수: {len(raw)}")

# 일자가 숫자가 아닌 행 제거 (합계행, 빈 행 등)
raw["일자"] = pd.to_numeric(raw["일자"], errors="coerce")
raw = raw.dropna(subset=["일자"]).copy()
raw["일자"] = raw["일자"].astype(int)

# 실제 날짜 만들기
def make_date(row):
    try:
        return pd.Timestamp(year=int(row["연도"]), month=int(row["월"]), day=int(row["일자"]))
    except Exception:
        return pd.NaT

raw["날짜"] = raw.apply(make_date, axis=1)
raw = raw.dropna(subset=["날짜"]).copy()

# 실입장객이 숫자가 아닌 행 제거 (휴장일 등)
raw["실입장객"] = pd.to_numeric(raw["실입장객"], errors="coerce")
before = len(raw)
raw = raw.dropna(subset=["실입장객"]).copy()
print(f"🧹 실입장객 결측 제거: {before} → {len(raw)}")

# 0 또는 음수 입장객은 휴장으로 간주, 제거
raw = raw[raw["실입장객"] > 0].copy()
print(f"🧹 휴장일(0) 제거 후: {len(raw)}")

# 기온/시간을 숫자로
for col in ["최저기온", "최고기온"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# 운영시간(시간) 계산: "10:30" 같은 문자열 → 분
def to_minutes(v):
    if pd.isna(v): return np.nan
    s = str(v)
    m = re.search(r"(\d{1,2})[:\s시](\d{0,2})", s)
    if not m: return np.nan
    h = int(m.group(1)); mn = int(m.group(2)) if m.group(2) else 0
    return h * 60 + mn

raw["개장_분"] = raw["개장시간"].apply(to_minutes)
raw["폐장_분"] = raw["폐장시간"].apply(to_minutes)
raw["운영시간_시간"] = (raw["폐장_분"] - raw["개장_분"]) / 60.0

# === 파생변수 ===
raw["요일num"] = raw["날짜"].dt.dayofweek         # 월=0, 일=6
raw["주말여부"] = (raw["요일num"] >= 5).astype(int)
raw["월_숫자"] = raw["날짜"].dt.month
raw["일"] = raw["날짜"].dt.day

# 공휴일
kr_holidays = holidays.country_holidays("KR", years=[2023, 2024, 2025])
raw["공휴일"] = raw["날짜"].dt.date.apply(lambda d: 1 if d in kr_holidays else 0)

# 방학 여부 (대략): 1-2월, 7-8월
raw["방학"] = raw["월_숫자"].isin([1, 2, 7, 8]).astype(int)

# 날씨 카테고리 단순화: 맑음/흐림/비/눈/기타
def simplify_weather(w):
    if pd.isna(w): return "기타"
    s = str(w)
    if "맑" in s or "쾌청" in s: return "맑음"
    if "눈" in s: return "눈"
    if "비" in s or "소나기" in s: return "비"
    if "흐림" in s or "구름" in s: return "흐림"
    return "기타"

raw["날씨_단순"] = raw["날씨"].apply(simplify_weather)

# 최종 컬럼 선택
final = raw[[
    "날짜", "연도", "월_숫자", "일", "요일num", "주말여부", "공휴일", "방학",
    "날씨_단순", "최저기온", "최고기온", "운영시간_시간", "실입장객"
]].rename(columns={
    "월_숫자": "월",
    "요일num": "요일",
    "날씨_단순": "날씨",
}).sort_values("날짜").reset_index(drop=True)

print(f"\n📊 최종 데이터:")
print(final.describe(include='all').to_string())
print(f"\n샘플:")
print(final.head(5).to_string())
print(f"\n결측치:")
print(final.isna().sum())

out_path = OUT_DIR / "seoulland_clean.csv"
final.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 저장됨: {out_path}")
