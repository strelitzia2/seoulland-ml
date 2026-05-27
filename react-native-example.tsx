/**
 * 서울랜드 입장객 예측 ML 클라이언트 (React Native)
 *
 * 메인 앱에서 이 파일 하나만 복사해서 쓰면 됩니다.
 * 메인 앱의 기존 코드는 일절 건드리지 않습니다.
 *
 * 사용법:
 *   import { predictCongestion } from './lib/visitorPredict';
 *   const result = await predictCongestion({ date: '2026-05-05', weather: '맑음' });
 *   if (result) { alert(result.message); }  // 실패 시 null 반환 — 앱 안 죽음
 */

// ============================================================================
// 환경별 서버 주소 설정
// ============================================================================
// 1) Railway 배포 후 (실제 배포):
const API_BASE = 'https://web-production-82f71.up.railway.app';
//
// 2) 로컬 개발 (uvicorn 실행 중):
//    iOS 시뮬레이터:    'http://localhost:8000'
//    Android 에뮬레이터: 'http://10.0.2.2:8000'
//    실제 폰(같은 WiFi):  'http://192.168.0.XX:8000'  (Mac의 IP)
//
// 한 곳에서만 바꾸면 전 앱에서 적용됩니다.

const FETCH_TIMEOUT_MS = 8000;   // 8초 타임아웃 (시연 시 UX 보호)


// ============================================================================
// 타입 정의 (서버 응답과 1:1 매칭)
// ============================================================================
export type Weather = '맑음' | '흐림' | '비' | '눈' | '안개' | '기타';
export type CongestionLevel = '한산' | '보통' | '붐빔' | '매우 붐빔';

export interface PredictInput {
  date: string;                     // 'YYYY-MM-DD' (필수)
  weather?: Weather;                // 기본 '맑음'
  min_temp?: number;                // 미입력 시 월 평년값
  max_temp?: number;                // 미입력 시 월 평년값
  precipitation_mm?: number;        // 기본 0
  operating_hours?: number;         // 기본 11
}

export interface PredictResult {
  date: string;
  predicted_visitors: number;       // 예: 7234
  average_visitors: number;         // 예: 4420 (3년 평균)
  congestion_ratio: number;         // 예: 1.66 (평균 대비 배수)
  congestion_percent: number;       // 예: 66 → 평균보다 +66%
  congestion_level: CongestionLevel;
  congestion_emoji: string;         // '🟢🟡🟠🔴'
  message: string;                  // '평균보다 66% 더 붐빌 예정'
  recommendation: string;           // '매우 붐빌 예정입니다. 공휴일+나들이 좋은 계절 영향.'
}


// ============================================================================
// 공통 fetch + 타임아웃 헬퍼
// ============================================================================
async function fetchWithTimeout(url: string, init?: RequestInit, ms = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}


// ============================================================================
// 1) 메인 함수: 혼잡도 예측
// ============================================================================
/**
 * 날짜+날씨를 받아 예상 입장객과 혼잡도를 반환.
 * 서버 오류/타임아웃 시 null 반환 → 호출 측에서 폴백 UI 처리.
 */
export async function predictCongestion(
  input: PredictInput
): Promise<PredictResult | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        date: input.date,
        weather: input.weather ?? '맑음',
        min_temp: input.min_temp ?? 15,
        max_temp: input.max_temp ?? 22,
        precipitation_mm: input.precipitation_mm ?? 0,
        operating_hours: input.operating_hours ?? 11,
      }),
    });
    if (!res.ok) {
      console.warn(`[predictCongestion] HTTP ${res.status}`);
      return null;
    }
    return (await res.json()) as PredictResult;
  } catch (e) {
    console.warn('[predictCongestion] error:', e);
    return null;     // 네트워크/타임아웃 — 앱 절대 죽지 않음
  }
}


// ============================================================================
// 2) 간단 GET 호출 (날씨 정보 없이 날짜만)
// ============================================================================
export async function predictCongestionByDate(
  date: string
): Promise<PredictResult | null> {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/predict?date=${encodeURIComponent(date)}`
    );
    if (!res.ok) return null;
    return (await res.json()) as PredictResult;
  } catch {
    return null;
  }
}


// ============================================================================
// 3) 통계 조회 (3년치 평균 등)
// ============================================================================
export interface SeoullandStats {
  data_period: { start: string; end: string; total_days: number };
  overall_mean: number;
  weekday_mean: number;
  weekend_mean: number;
  holiday_mean: number;
  by_weekday: Record<string, number>;
  by_month: Record<string, number>;
  by_weather: Record<string, number>;
}

export async function getStats(): Promise<SeoullandStats | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/stats`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}


// ============================================================================
// 4) UI 헬퍼: 혼잡도 색상
// ============================================================================
export function congestionColor(level: CongestionLevel): string {
  switch (level) {
    case '한산':     return '#10B981';   // green-500
    case '보통':     return '#F59E0B';   // amber-500
    case '붐빔':     return '#F97316';   // orange-500
    case '매우 붐빔': return '#EF4444';   // red-500
  }
}


// ============================================================================
// 5) 사용 예시 (React Native 컴포넌트)
// ============================================================================
/*
import React, { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import {
  predictCongestion, congestionColor, PredictResult,
} from './lib/visitorPredict';

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
  if (!result) return null;  // 서버 오류 시 카드 자체를 숨김 (안전)

  return (
    <View style={{
      padding: 16, borderRadius: 12,
      backgroundColor: congestionColor(result.congestion_level) + '20',
      borderColor: congestionColor(result.congestion_level), borderWidth: 1,
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
*/
