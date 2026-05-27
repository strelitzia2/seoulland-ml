"""
08_preprocess_v2.py - 6개 xlsx 통합 (기상청 데이터 추가)

기존 02_preprocess.py에 기상청 파일 3개를 합쳐 변수 추가:
- 평균기온 (기존엔 최저/최고만)
- 일강수량(mm) ← 강력한 신호
- 일조시간 (일출~일몰 분 단위)
- 날씨(기상청) ← 별도 카테고리

출력: data/seoulland_clean_v2.csv
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
import holidays

DATASETS = Path.home() / "Desktop" / "Datasets"
SEOULLAND_DIR = DATASETS / "1.일별입장객및기상조건"
KMA_DIR = DATASETS / "3.기상_서울관측소기준"
OUT_DIR = Path.home() / "Desktop" / "seoulland-ml" / "data"

# ===== 1. 서울랜드 내부 데이터 (기존 로직) =====
COLS = ["일자", "요일_str", "날씨_내부", "최저기온_내부", "최고기온_내부",
        "개장시간", "폐장시간", "매표입장객_주간", "실입장객"]
rows = []
for xlsx in sorted(SEOULLAND_DIR.glob("*.xlsx")):
    year = 2000 + int(re.search(r"(\d{2})", xlsx.name).group(1))
    for sheet in pd.ExcelFile(xlsx).sheet_names:
        month = int(re.search(r"(\d+)", sheet).group(1))
        df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=3, header=None, names=COLS)
        df["연도"] = year; df["월"] = month
        rows.append(df)
seoul = pd.concat(rows, ignore_index=True)
seoul["일자"] = pd.to_numeric(seoul["일자"], errors="coerce")
seoul = seoul.dropna(subset=["일자"]).copy()
seoul["일자"] = seoul["일자"].astype(int)
seoul["날짜"] = seoul.apply(
    lambda r: pd.Timestamp(year=int(r["연도"]), month=int(r["월"]), day=int(r["일자"]))
    if 1 <= r["일자"] <= 31 else pd.NaT, axis=1)
seoul = seoul.dropna(subset=["날짜"]).copy()
seoul["실입장객"] = pd.to_numeric(seoul["실입장객"], errors="coerce")
seoul = seoul.dropna(subset=["실입장객"])
seoul = seoul[seoul["실입장객"] > 0].copy()
for c in ["최저기온_내부", "최고기온_내부"]:
    seoul[c] = pd.to_numeric(seoul[c], errors="coerce")

def to_min(v):
    if pd.isna(v): return np.nan
    m = re.search(r"(\d{1,2})[:\s시](\d{0,2})", str(v))
    if not m: return np.nan
    return int(m.group(1))*60 + (int(m.group(2)) if m.group(2) else 0)

seoul["개장_분"] = seoul["개장시간"].apply(to_min)
seoul["폐장_분"] = seoul["폐장시간"].apply(to_min)
seoul["운영시간_시간"] = (seoul["폐장_분"] - seoul["개장_분"]) / 60
print(f"📁 서울랜드: {len(seoul)}행")

# ===== 2. 기상청 데이터 통합 =====
KMA_COLS = ["_skip", "구분", "일출시간", "일몰시간", "평균기온", "최고기온", "최저기온",
            "일강수량", "날씨_기상청", "날씨_내부관리"]
kma_rows = []
for xlsx in sorted(KMA_DIR.glob("*.xlsx")):
    for sheet in pd.ExcelFile(xlsx).sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=3, header=None, names=KMA_COLS)
        kma_rows.append(df)
kma = pd.concat(kma_rows, ignore_index=True)
kma["날짜"] = pd.to_datetime(kma["구분"], errors="coerce")
kma = kma.dropna(subset=["날짜"]).copy()
# 수치 변환
for c in ["평균기온", "최고기온", "최저기온", "일강수량"]:
    kma[c] = pd.to_numeric(kma[c], errors="coerce")
# 일강수량 결측 = 0mm (안 옴)
kma["일강수량"] = kma["일강수량"].fillna(0)
# 일조시간(분): 일출~일몰
kma["일출_분"] = kma["일출시간"].apply(to_min)
kma["일몰_분"] = kma["일몰시간"].apply(to_min)
kma["일조시간_시간"] = (kma["일몰_분"] - kma["일출_분"]) / 60
print(f"📁 기상청: {len(kma)}행")

# ===== 3. 날짜 기준 병합 =====
merged = seoul.merge(
    kma[["날짜", "평균기온", "최고기온", "최저기온", "일강수량",
         "일조시간_시간", "날씨_기상청"]],
    on="날짜", how="left", suffixes=("_seoul", "_kma"))
print(f"🔗 병합 후: {len(merged)}행")

# 기상청 데이터 결측 = 우리 매칭 안 된 날짜. 폴백: 내부 데이터 사용
merged["최저기온"] = merged["최저기온"].fillna(merged["최저기온_내부"])
merged["최고기온"] = merged["최고기온"].fillna(merged["최고기온_내부"])
merged["평균기온"] = merged["평균기온"].fillna(
    (merged["최저기온"] + merged["최고기온"]) / 2)
merged["일강수량"] = merged["일강수량"].fillna(0)
merged["일조시간_시간"] = merged["일조시간_시간"].fillna(11)  # 평균값
merged["날씨_기상청"] = merged["날씨_기상청"].fillna("정보없음")

print(f"   기상청 매칭률: {(~merged['일강수량'].isna()).sum()}/{len(merged)}")
print(f"   평균 강수량: {merged['일강수량'].mean():.1f}mm")
print(f"   비온 날(>0mm): {(merged['일강수량']>0).sum()}일")

# ===== 4. 파생변수 =====
merged["요일"] = merged["날짜"].dt.dayofweek
merged["주말여부"] = (merged["요일"] >= 5).astype(int)
merged["월_num"] = merged["날짜"].dt.month
merged["일"] = merged["날짜"].dt.day
kr_h = holidays.country_holidays("KR", years=[2023, 2024, 2025])
merged["공휴일"] = merged["날짜"].dt.date.apply(lambda d: 1 if d in kr_h else 0)
merged["방학"] = merged["월_num"].isin([1, 2, 7, 8]).astype(int)

# 강수 카테고리 (비/눈/맑음 모델이 일강수량과 함께 보면 더 강함)
merged["강수여부"] = (merged["일강수량"] > 0).astype(int)
merged["폭우"] = (merged["일강수량"] > 30).astype(int)

# 내부 날씨 단순화 (기존 로직)
def simplify(w):
    if pd.isna(w): return "기타"
    s = str(w)
    if "맑" in s or "쾌청" in s: return "맑음"
    if "눈" in s: return "눈"
    if "비" in s or "소나기" in s: return "비"
    if "흐림" in s or "구름" in s: return "흐림"
    return "기타"
merged["날씨"] = merged["날씨_내부"].apply(simplify)

# 기상청 날씨 단순화
def simplify_kma(w):
    if pd.isna(w): return "기타"
    s = str(w)
    if "맑" in s: return "맑음"
    if "눈" in s: return "눈"
    if "비" in s or "소나기" in s or "뇌우" in s: return "비"
    if "흐림" in s or "구름많음" in s: return "흐림"
    if "연무" in s or "박무" in s or "안개" in s: return "안개"
    return "기타"
merged["날씨_kma_단순"] = merged["날씨_기상청"].apply(simplify_kma)

# ===== 5. 최종 출력 =====
final = merged[[
    "날짜", "연도", "월_num", "일", "요일", "주말여부", "공휴일", "방학",
    "날씨", "날씨_kma_단순",
    "평균기온", "최저기온", "최고기온",
    "일강수량", "강수여부", "폭우",
    "일조시간_시간", "운영시간_시간",
    "실입장객"
]].rename(columns={"월_num": "월"}).dropna(subset=["운영시간_시간"]).sort_values("날짜").reset_index(drop=True)

print(f"\n📊 최종: {len(final)}행, {final.shape[1]}컬럼")
print(f"\n새 변수 통계:")
print(final[["평균기온", "일강수량", "강수여부", "일조시간_시간"]].describe().to_string())
print(f"\n결측치:")
print(final.isna().sum()[final.isna().sum() > 0])

out = OUT_DIR / "seoulland_clean_v2.csv"
final.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n✅ 저장: {out}")
