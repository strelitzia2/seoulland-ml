"""기상청 데이터 구조 확인"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "Desktop" / "Datasets" / "3.기상_서울관측소기준"

for xlsx in sorted(DATA_DIR.glob("*.xlsx")):
    print(f"\n📄 {xlsx.name}")
    xl = pd.ExcelFile(xlsx)
    print(f"   시트: {xl.sheet_names}")
    # 첫 시트만 자세히
    first_sheet = xl.sheet_names[0]
    # 헤더 구조 확인을 위해 첫 5행 raw로
    raw = pd.read_excel(xlsx, sheet_name=first_sheet, header=None, nrows=6)
    print(f"   ── 첫 시트 '{first_sheet}' 상위 6행 ──")
    print(raw.to_string(max_colwidth=15))
    break  # 한 파일만
