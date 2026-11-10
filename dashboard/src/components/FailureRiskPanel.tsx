import type { FailureRiskResponse } from '../api'
import StatusBadge from './StatusBadge'

interface Props {
    data: FailureRiskResponse | null
    loading: boolean
    error: boolean
}

function probColor(p: number): string {
    if (p > 0.5) return 'var(--clr-danger)'
    if (p > 0.2) return 'var(--clr-warning)'
    return 'var(--clr-success)'
}

export default function FailureRiskPanel({ data, loading, error }: Props) {
    const probs = data?.probabilities ?? null

    return (
        <div className="panel col-6">
            <div className="panel-header">
                <span className="panel-title">
                    <span>⚡</span> Failure Risk
                </span>
                {data && <StatusBadge status={data.status} />}
            </div>

            {loading && !data && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {[...Array(4)].map((_, i) => <div className="skeleton" key={i} style={{ height: 24 }} />)}
                </div>
            )}

            {error && !data && (
                <div className="state-message">
                    <span className="state-message-icon">⚠️</span>
                    <span className="state-message-title">Could not load failure risk</span>
                </div>
            )}

            {data && !probs && (
                <div className="state-message">
                    <span className="state-message-icon">🏗️</span>
                    <span className="state-message-title">
                        {data.status === 'not_trained' ? 'Model not trained' : 'Insufficient data'}
                    </span>
                    <span className="state-message-sub">{data.message}</span>
                    {data.status === 'not_trained' && (
                        <span className="state-message-sub" style={{ marginTop: 8 }}>
                            Requires verified <code style={{ color: 'var(--clr-accent)', fontFamily: 'var(--font-mono)' }}>FailureEvent</code> rows.<br />
                            See README → Stage B.
                        </span>
                    )}
                </div>
            )}

            {data?.status === 'prediction_available' && probs && (
                <>
                    <div className="prob-bar-wrap">
                        {Object.entries(probs).map(([key, val]) => {
                            const horizon = key.replace('p', '').replace('h', '') + 'h'
                            const pct = Math.round(val * 100)
                            return (
                                <div className="prob-row" key={key}>
                                    <span className="prob-label">{horizon}</span>
                                    <div className="prob-track">
                                        <div
                                            className="prob-fill"
                                            style={{ width: `${pct}%`, background: probColor(val) }}
                                        />
                                    </div>
                                    <span className="prob-val" style={{ color: probColor(val) }}>
                                        {pct}%
                                    </span>
                                </div>
                            )
                        })}
                    </div>
                    <div className="disclaimer" style={{ marginTop: 'var(--space-4)' }}>
                        ⚠️ Calibrated on held-out runs. Not a guarantee of failure. Consult a maintenance engineer before acting.
                    </div>
                </>
            )}
        </div>
    )
}
