# 📱 React Native 앱 ↔ ML 서버 연결 가이드

> 메인 앱(셔틀버스 React Native)에서 서울랜드 입장객 예측 ML 서버를 호출하는 방법.
> **메인 앱 코드는 일절 건드리지 않습니다.** 새 파일 1개 추가 + 사용할 화면에서 import만 하면 끝.

---

## 🗺️ 전체 구조

```
┌────────────────────────────┐         ┌────────────────────────────┐
│  React Native 메인 앱       │         │  ML 서버 (Railway)          │
│  (셔틀버스 캡스톤 프로젝트)  │  fetch  │  https://web-production-   │
│                            │ ──────→ │  82f71.up.railway.app       │
│  ┌──────────────────────┐  │         │                             │
│  │ lib/visitorPredict   │  │ ←────── │  /predict (POST/GET)        │
│  │  .ts (새 파일 1개)   │  │   JSON  │  /stats                     │
│  └──────────────────────┘  │         │  /docs (API 문서)           │
│       ↓ import              │         │                             │
│  ┌──────────────────────┐  │         │  XGBoost 모델 로드          │
│  │ 예약 화면 / 챗봇 등  │  │         │  R² = 0.798                 │
│  │  (기존 화면)         │  │         │                             │
│  └──────────────────────┘  │         └────────────────────────────┘
└────────────────────────────┘
```

**핵심 원칙:**
- 메인 앱은 ML 서버 존재를 **거의 모름**. fetch 한 번이면 끝.
- 서버 죽어도 메인 앱 **절대 안 죽음** (null 폴백).
- 메인 앱 기존 코드 **0줄 수정**.

---

## ✅ 사전 확인

| 항목 | 확인 |
|---|---|
| ML 서버 URL 작동 | https://web-production-82f71.up.railway.app/docs 브라우저로 열어서 Swagger UI 보이면 OK |
| 메인 앱이 React Native | TypeScript 환경 권장 (JavaScript도 가능, 타입만 빼면 됨) |
| 메인 앱이 fetch 사용 가능 | React Native는 기본 내장. 별도 설치 X |
| 추가 라이브러리 | **없음**. axios나 다른 패키지 설치 불필요. |

---

## 📂 STEP 1 — 클라이언트 파일 복사 (30초)

ML 클라이언트 파일을 메인 앱 폴더에 복사:

```bash
# 메인 앱 폴더에서 (예: ~/Desktop/shuttle-app)
mkdir -p lib
cp ~/Desktop/seoulland-ml/react-native-example.tsx ./lib/visitorPredict.ts
```

이 파일에는 다음이 포함되어 있어요:
- `predictCongestion()` 함수 (메인)
- `predictCongestionByDate()` 함수 (간단 호출)
- `getStats()` 함수 (3년 통계)
- `congestionColor()` 헬퍼 (UI 색상)
- `CongestionCard` 컴포넌트 예시 (복붙용)

**중요:** 파일 안의 `API_BASE` 상수가 이미 Railway URL로 설정되어 있어요. 추가 수정 불필요.

```typescript
const API_BASE = 'https://web-production-82f71.up.railway.app';
```

---

## 💻 STEP 2 — 간단한 호출 (최소 코드)

어떤 화면이든 import 한 줄로 사용:

```typescript
import { predictCongestion } from '@/lib/visitorPredict';

// 어디서든 호출 가능 (useEffect, 버튼 onPress 등)
const result = await predictCongestion({
  date: '2026-05-05',     // YYYY-MM-DD (필수)
  weather: '맑음',         // 선택, 기본 '맑음'
});

if (result) {
  console.log(result.message);              // "평균보다 49% 더 붐빌 예정"
  console.log(result.predicted_visitors);   // 6592
  console.log(result.congestion_level);     // "붐빔"
}
// result === null 이면 서버 오류 → 폴백 UI
```

### 응답 객체 전체 필드

```typescript
{
  date: "2026-05-05",
  predicted_visitors: 6592,        // 예상 입장객 수
  average_visitors: 4420,          // 3년 평균
  congestion_ratio: 1.49,          // 평균 대비 배수
  congestion_percent: 49,          // 평균보다 +49% (음수면 -)
  congestion_level: "붐빔",        // 한산/보통/붐빔/매우 붐빔
  congestion_emoji: "🟠",          // 🟢🟡🟠🔴
  message: "평균보다 49% 더 붐빌 예정",
  recommendation: "평소보다 붐빌 예정입니다. 공휴일 + 나들이 좋은 계절 영향."
}
```

---

## 🎨 STEP 3 — UI 카드 컴포넌트 (예약 화면용)

`lib/visitorPredict.ts` 파일 하단에 `CongestionCard` 예시가 있어요. 그걸 그대로 복사해서 새 컴포넌트로 만들거나, 아래 코드를 예약 화면에 직접 붙여넣어도 됩니다.

### 새 파일로 분리하는 방법

```typescript
// components/CongestionCard.tsx
import React, { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { predictCongestion, congestionColor, PredictResult } from '@/lib/visitorPredict';

export function CongestionCard({ date }: { date: string }) {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    predictCongestion({ date, weather: '맑음' })
      .then(setResult)
      .finally(() => setLoading(false));
  }, [date]);

  if (loading) return <ActivityIndicator />;
  if (!result) return null;   // 서버 오류 시 카드 자체를 숨김

  const color = congestionColor(result.congestion_level);

  return (
    <View style={{
      padding: 16,
      borderRadius: 12,
      backgroundColor: color + '20',   // 투명도 20%
      borderColor: color,
      borderWidth: 1,
    }}>
      <Text style={{ fontSize: 18, fontWeight: '700' }}>
        🎢 {result.congestion_emoji} {result.congestion_level}
      </Text>
      <Text style={{ fontSize: 14, marginTop: 4 }}>
        예상 {result.predicted_visitors.toLocaleString()}명 · {result.message}
      </Text>
      <Text style={{ fontSize: 12, marginTop: 8, color: '#666' }}>
        {result.recommendation}
      </Text>
    </View>
  );
}
```

### 예약 화면에서 사용

```typescript
// app/booking.tsx (또는 사용하시는 예약 화면)
import { CongestionCard } from '@/components/CongestionCard';

export function BookingScreen() {
  const [selectedDate, setSelectedDate] = useState('2026-05-05');

  return (
    <ScrollView>
      {/* 기존 캘린더 코드 ... */}
      <Calendar onDayPress={(day) => setSelectedDate(day.dateString)} />

      {/* ↓ 한 줄 추가만 하면 끝 */}
      <CongestionCard date={selectedDate} />

      {/* 기존 코드 계속 ... */}
    </ScrollView>
  );
}
```

---

## 🤖 STEP 4 (선택) — AI 챗봇 연동

챗봇이 사용자 질문을 받으면 자동으로 ML 서버 호출:

```typescript
// 챗봇 응답 생성 함수 안에서
import { predictCongestion } from '@/lib/visitorPredict';

async function handleUserMessage(text: string) {
  // 간단한 패턴 매칭: 날짜 + "붐비"  키워드
  const dateMatch = text.match(/(\d{4}-\d{2}-\d{2})/);
  const wantsCongestion = /붐비|혼잡|사람|얼마나/.test(text);

  if (dateMatch && wantsCongestion) {
    const r = await predictCongestion({ date: dateMatch[1] });
    if (r) {
      return `${dateMatch[1]}은 ${r.message}. ${r.recommendation}`;
    }
  }

  // 기존 챗봇 응답 로직 ...
}
```

---

## 🔧 트러블슈팅

### ❌ "Network request failed"

**원인:** 네트워크 연결 문제 또는 CORS.

**해결:**
1. 브라우저로 `https://web-production-82f71.up.railway.app/` 열어서 응답 보이는지 확인
2. CORS는 서버에서 `*` 허용이라 문제 없음
3. iOS 시뮬레이터의 경우 Mac이 인터넷 연결되어 있는지 확인

### ❌ 응답은 오는데 한글이 깨짐

**원인:** 거의 없음. fetch가 자동으로 UTF-8 처리.

**해결:** `res.json()` 대신 `JSON.parse(await res.text())` 시도 — 하지만 거의 안 일어남.

### ❌ Railway 서버가 잠들어 있어서 첫 호출이 느림 (cold start)

**증상:** 첫 호출만 5~30초 걸리고 이후엔 빠름.

**해결:**
- 무료 티어 특성. 정상.
- 시연 직전에 한 번 호출해서 깨워두면 OK
- 또는 앱 시작 시 백그라운드로 `getStats()` 한 번 호출 (워밍업)

```typescript
// App.tsx (앱 진입 시)
useEffect(() => {
  getStats();   // 결과는 안 써도 됨. 서버 깨우는 용도
}, []);
```

### ❌ TypeScript 에러: "Cannot find module '@/lib/visitorPredict'"

**원인:** 경로 alias 설정.

**해결:** `tsconfig.json`에 다음이 없으면 상대 경로로:
```typescript
import { predictCongestion } from '../lib/visitorPredict';
```

### ⚠️ Railway 서비스 다운 (드물게)

**자동 처리:** 클라이언트가 `null` 반환 → `CongestionCard`가 `null`이면 자체적으로 카드 숨김. 메인 앱 정상 작동.

**확인:** `https://railway.app/dashboard` → 프로젝트 → Deployments 상태 확인.

---

## 🎬 시연 체크리스트

발표 1시간 전:

- [ ] **워밍업**: 브라우저로 `/predict?date=오늘날짜` 한 번 호출 → cold start 방지
- [ ] **메인 앱 빌드**: `npx expo start` (또는 `npm start`)
- [ ] **시뮬레이터 실행**: iOS/Android
- [ ] **테스트 호출**: 예약 화면에서 날짜 바꿔보며 혼잡도 카드 표시 확인
- [ ] **오프라인 시나리오 테스트**: WiFi 끄고 카드가 어떻게 보이는지 (자연스럽게 숨겨져야 함)

발표 중:

- [ ] 인터넷 연결 확인 (시연용 와이파이 또는 핫스팟)
- [ ] 백업: 만약 Railway 꺼지면 → 로컬 서버 띄울 수 있도록 노트북 준비
  ```bash
  cd ~/Desktop/seoulland-ml && source venv/bin/activate
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
  → `lib/visitorPredict.ts`의 `API_BASE`를 `http://<Mac IP>:8000`으로 임시 교체

---

## 📊 발표 시 강조 포인트

> "본 셔틀버스 앱에는 별도로 개발한 머신러닝 기반 혼잡도 예측 시스템이 통합되어 있습니다.
> 사용자가 예약 날짜를 선택하면 React Native 앱에서 자동으로 Railway에 배포된 ML 서버를 호출하고,
> 서버는 3년치 서울랜드 운영 데이터와 기상청 데이터를 학습한 XGBoost 모델로 예상 입장객 수와
> 평균 대비 혼잡도를 반환합니다. 본 시스템은 메인 앱과 완전히 분리된 독립 서비스로 운영되어
> 메인 앱의 안정성에 영향을 주지 않으며, 서버 장애 시 자동으로 폴백되도록 설계되었습니다."

**키워드:**
- 머신러닝 (XGBoost, R² 0.798)
- REST API (FastAPI)
- 클라우드 배포 (Railway)
- 서비스 분리 (Microservice-like)
- 장애 대응 (Graceful degradation)

---

## 📞 참고 URL

- **API 라이브 문서**: https://web-production-82f71.up.railway.app/docs
- **GitHub 리포**: https://github.com/strelitzia2/seoulland-ml
- **Railway 대시보드**: https://railway.app/dashboard

---

## 끝

연동 자체는 **fetch 한 번**이면 끝나는 매우 단순한 구조입니다.
이 문서를 옆에 두고 STEP 1~3만 따라하시면 30분 안에 끝나요. 🎢
