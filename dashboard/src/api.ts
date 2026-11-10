/* Typed API helpers for the PM dashboard */

const API = (import.meta as any).env?.VITE_API ?? ''

async function get<T>(path: string): Promise<T> {
    const res = await fetch(`${API}${path}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json() as Promise<T>
}

// ---------- Types ----------

export type ApiStatus =
    | 'ok'
    | 'no_data'
    | 'not_trained'
    | 'insufficient_data'
    | 'prediction_available'
    | 'model_unavailable'

export interface Observation {
    data_item_id: string
    name: string
    type: string
    category: string
    value: number | string | null
    unavailable: boolean
    units: string | null
    timestamp: string
    condition_level: string | null
    component_path: string
}

export interface LiveResponse {
    status: ApiStatus
    machine_id: string | null
    observations: Observation[]
    generated_at: string
}

export interface AnomalyResult {
    timestamp: string
    operating_state: string
    anomaly_score: number | null
    status: string
    reason: string | null
    top_features: { feature: string; z: number }[]
}

export interface AnomalyResponse {
    machine_id: string | null
    status: ApiStatus
    kind?: string
    disclaimer?: string | null
    results?: AnomalyResult[]
    message?: string
    generated_at: string
}

export interface FailureRiskResponse {
    machine_id: string | null
    status: ApiStatus
    horizons_hours?: number[]
    probabilities: Record<string, number> | null
    message?: string
    generated_at: string
}

export interface RULResponse {
    machine_id: string | null
    status: ApiStatus
    rul_hours: number | null
    message?: string
    generated_at: string
}

export interface Recommendation {
    signal_type: string
    urgency: string
    risk_level: string
    detected_issue: string
    suggested_inspection: string[]
    confidence: string
    limitations: string
}

export interface RecommendationsResponse {
    machine_id: string | null
    status: ApiStatus
    recommendations: Recommendation[]
    note?: string
    message?: string
    generated_at: string
}

export interface StageStatus {
    status: string
    artifact_present: boolean
    trained_at: string | null
    detail?: Record<string, unknown> | null
}

export interface ModelStatusResponse {
    stages: Record<string, StageStatus>
    generated_at: string
}

// ---------- Fetchers ----------

export const fetchLive = () => get<LiveResponse>('/api/machine/live')
export const fetchAnomalies = () => get<AnomalyResponse>('/api/predictions/anomalies')
export const fetchFailureRisk = () => get<FailureRiskResponse>('/api/predictions/failure-risk')
export const fetchRUL = () => get<RULResponse>('/api/predictions/rul')
export const fetchRecommendations = () => get<RecommendationsResponse>('/api/maintenance/recommendations')
export const fetchModelStatus = () => get<ModelStatusResponse>('/api/model/status')
export const fetchHealth = () => get<{ ok: boolean }>('/health')
