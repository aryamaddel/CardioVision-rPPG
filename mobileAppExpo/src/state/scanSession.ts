import type { RPPGResult } from '../api/rppgService';

export interface ScanSession {
  result: RPPGResult | null;
  videoUri?: string;
}

let currentSession: ScanSession = {
  result: null,
};

export function setScanSession(session: ScanSession) {
  currentSession = session;
}

export function getScanSession(): ScanSession {
  return currentSession;
}

export function clearScanSession() {
  currentSession = { result: null };
}
