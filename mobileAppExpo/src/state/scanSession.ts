import type { RPPGResult } from '../api/rppgService';

export interface ScanSession {
  result: RPPGResult | null;
  videoUri?: string;
}

let currentSession: ScanSession = {
  result: null,
};
let pendingResultPromise: Promise<RPPGResult> | null = null;

export function setScanSession(session: ScanSession) {
  currentSession = session;
}

export function getScanSession(): ScanSession {
  return currentSession;
}

export function clearScanSession() {
  currentSession = { result: null };
  pendingResultPromise = null;
}

export function setPendingScanResult(promise: Promise<RPPGResult> | null) {
  pendingResultPromise = promise;
}

export function getPendingScanResult(): Promise<RPPGResult> | null {
  return pendingResultPromise;
}

export function clearPendingScanResult() {
  pendingResultPromise = null;
}
