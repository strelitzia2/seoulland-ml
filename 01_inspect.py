"""
01_inspect.py - 데이터 파일 구조 점검

각 xlsx 파일이 어떤 컬럼을 가지고 있는지, 행 수는 몇 개인지,
어떤 데이터가 들어있는지 확인합니다.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "Desktop" / "Datasets"

folders = {
    "입장객+기상": DATA_DIR / "1.일별입장객및기상조건",
    "기상청": DATA_DIR / "3.기상_서울관측소기준",
}

for label, folder in folders.items():
    print(f"\n{'='*70}")
    print(f"📁 {label}: {folder.name}")
    print('='*70)
    for xlsx in sorted(folder.glob("*.xlsx")):
        print(f"\n📄 {xlsx.name}")
        # 모든 시트 이름
        xl = pd.ExcelFile(xlsx)
        print(f"   시트: {xl.sheet_names}")
        for sheet in xl.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet)
            print(f"\n   ─── 시트 '{sheet}' ───")
            print(f"   행 수: {len(df)}, 열 수: {len(df.columns)}")
            print(f"   컬럼: {list(df.columns)}")
            print(f"   첫 3행:")
            print(df.head(3).to_string(max_colwidth=20))
