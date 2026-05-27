# 서울랜드 입장객 예측 ML

## 폴더 구조
```
seoulland-ml/
├── venv/                       # Python 가상환경 (Git에 안 올림)
├── data/seoulland_clean.csv    # 전처리된 학습 데이터
├── models/
│   ├── model.pkl               # 학습된 XGBoost 모델
│   └── metadata.pkl            # 평균/특성 목록 등
├── api/main.py                 # FastAPI 서버
├── 01_inspect.py               # 원본 데이터 점검
├── 02_preprocess.py            # 데이터 통합/전처리
├── 03_train.py                 # 모델 학습
├── predict.py                  # 예측 함수
├── react-native-example.tsx    # RN 앱에서 호출하는 예제
├── requirements.txt            # Python 패키지 목록 (배포용)
├── Procfile                    # Railway 실행 명령
└── runtime.txt                 # Python 버전 명시
```

## 다시 실행하려면

```bash
cd ~/Desktop/seoulland-ml
source venv/bin/activate          # 가상환경 진입 (프롬프트에 (venv) 표시됨)

# 데이터 전처리 → 모델 학습 (데이터가 바뀌었을 때만)
python 02_preprocess.py
python 03_train.py

# API 서버 실행
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs 열면 API 테스트 UI
```

## 모델 성능 (2025년 검증 결과)
- MAE: 1,241명
- MAPE: 45.8%
- R²: 0.753

## Railway 배포 (무료)

1. https://railway.app 에서 GitHub 로그인
2. 이 폴더(`seoulland-ml`)를 GitHub 리포지토리로 푸시
   ```bash
   cd ~/Desktop/seoulland-ml
   git init
   git add .
   git commit -m "initial"
   # GitHub에서 빈 리포 만들고:
   git remote add origin https://github.com/<유저명>/seoulland-ml.git
   git push -u origin main
   ```
3. Railway → "Deploy from GitHub repo" → 이 리포 선택
4. 자동으로 `requirements.txt` 읽고 `Procfile` 실행
5. 5분 후 `https://xxx.up.railway.app` URL 발급됨 → RN 앱의 `API_BASE` 에 붙여넣기

## React Native 연동
- `react-native-example.tsx` 파일을 RN 앱에 복사
- `API_BASE` 변수만 Railway URL로 바꾸면 끝
