// React Native에서 ML 서버 호출 예제
// 메인 앱에 이 함수만 추가하면 됩니다 (기존 코드 건드릴 필요 없음)

// === 환경별 서버 주소 ===
// 1) 로컬 테스트 (Mac에서 uvicorn 실행 중):
//    iOS 시뮬레이터: http://localhost:8000
//    Android 에뮬레이터: http://10.0.2.2:8000
//    실제 폰: http://<Mac의-IP>:8000  (예: http://192.168.0.10:8000)
//
// 2) Railway 배포 후:
//    https://your-app.up.railway.app
const API_BASE = 'http://localhost:8000';

export interface PredictResult {
  date: string;
  predicted_visitors: number;
  average_visitors: number;
  congestion_ratio: number;
  congestion_percent: number;   // 음수면 평균보다 덜 붐빔
  message: string;              // "평균보다 30% 더 붐빌 예정"
}

export interface PredictInput {
  date: string;                 // 'YYYY-MM-DD'
  weather: '맑음' | '흐림' | '비' | '눈' | '기타';
  min_temp: number;
  max_temp: number;
  operating_hours?: number;     // 기본 11
}

export async function predictCongestion(input: PredictInput): Promise<PredictResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(`Predict API error: ${res.status}`);
  return res.json();
}

// === 사용 예시 ===
// const result = await predictCongestion({
//   date: '2026-05-05',
//   weather: '맑음',
//   min_temp: 15,
//   max_temp: 24,
// });
// console.log(result.message);   // "평균보다 130% 더 붐빌 예정"
// console.log(result.predicted_visitors);  // 9997
