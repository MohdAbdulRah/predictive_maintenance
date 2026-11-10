import { useCallback, useEffect, useRef, useState } from 'react'
import Header from './components/Header'
import MachineStatus from './components/MachineStatus'
import AnomalyPanel from './components/AnomalyPanel'
import FailureRiskPanel from './components/FailureRiskPanel'
import RULPanel from './components/RULPanel'
import RecommendationsPanel from './components/RecommendationsPanel'
import ModelStatusPanel from './components/ModelStatusPanel'
import {
    fetchHealth,
    fetchLive,
    fetchAnomalies,
    fetchFailureRisk,
    fetchRUL,
    fetchRecommendations,
    fetchModelStatus,
} from './api'
import type {
    LiveResponse,
    AnomalyResponse,
    FailureRiskResponse,
    RULResponse,
    RecommendationsResponse,
    ModelStatusResponse,
} from './api'

const POLL_MS = 3000

export default function App() {
    const [connected, setConnected] = useState(false)
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
    const [machineId, setMachineId] = useState<string | null>(null)

    const [live, setLive] = useState<LiveResponse | null>(null)
    const [anomaly, setAnomaly] = useState<AnomalyResponse | null>(null)
    const [failure, setFailure] = useState<FailureRiskResponse | null>(null)
    const [rul, setRul] = useState<RULResponse | null>(null)
    const [recs, setRecs] = useState<RecommendationsResponse | null>(null)
    const [model, setModel] = useState<ModelStatusResponse | null>(null)

    const [liveErr, setLiveErr] = useState(false)
    const [anomalyErr, setAnomalyErr] = useState(false)
    const [failureErr, setFailureErr] = useState(false)
    const [rulErr, setRulErr] = useState(false)
    const [recsErr, setRecsErr] = useState(false)
    const [modelErr, setModelErr] = useState(false)

    const [loading, setLoading] = useState(true)
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const poll = useCallback(async () => {
        try {
            await fetchHealth()
            setConnected(true)
        } catch {
            setConnected(false)
        }

        await Promise.allSettled([
            fetchLive().then(d => { setLive(d); setMachineId(d.machine_id); setLiveErr(false) })
                .catch(() => setLiveErr(true)),
            fetchAnomalies().then(d => { setAnomaly(d); setAnomalyErr(false) })
                .catch(() => setAnomalyErr(true)),
            fetchFailureRisk().then(d => { setFailure(d); setFailureErr(false) })
                .catch(() => setFailureErr(true)),
            fetchRUL().then(d => { setRul(d); setRulErr(false) })
                .catch(() => setRulErr(true)),
            fetchRecommendations().then(d => { setRecs(d); setRecsErr(false) })
                .catch(() => setRecsErr(true)),
            fetchModelStatus().then(d => { setModel(d); setModelErr(false) })
                .catch(() => setModelErr(true)),
        ])

        setLastUpdated(new Date())
        setLoading(false)
        timerRef.current = setTimeout(poll, POLL_MS)
    }, [])

    useEffect(() => {
        poll()
        return () => { if (timerRef.current) clearTimeout(timerRef.current) }
    }, [poll])

    return (
        <div className="app-container">
            <Header connected={connected} lastUpdated={lastUpdated} machineId={machineId} />

            <main className="dashboard-grid" id="main-content">
                {/* Row 1: Machine status — full width */}
                <MachineStatus data={live} loading={loading} error={liveErr} />

                {/* Row 2: Anomaly + Failure side by side */}
                <AnomalyPanel data={anomaly} loading={loading} error={anomalyErr} />
                <FailureRiskPanel data={failure} loading={loading} error={failureErr} />

                {/* Row 3: Recommendations (wide) + RUL (narrow) */}
                <RecommendationsPanel data={recs} loading={loading} error={recsErr} />
                <RULPanel data={rul} loading={loading} error={rulErr} />

                {/* Row 4: Model status — full width */}
                <ModelStatusPanel data={model} loading={loading} error={modelErr} />
            </main>
        </div>
    )
}
