import type { RecommendationsResponse } from '../api'
import StatusBadge from './StatusBadge'

interface Props {
    data: RecommendationsResponse | null
    loading: boolean
    error: boolean
}

const urgencyOrder: Record<string, number> = { high: 0, medium: 1, low: 2 }

export default function RecommendationsPanel({ data, loading, error }: Props) {
    const recs = (data?.recommendations ?? []).slice().sort(
        (a, b) => (urgencyOrder[a.urgency] ?? 3) - (urgencyOrder[b.urgency] ?? 3)
    )

    return (
        <div className="panel col-8">
            <div className="panel-header">
                <span className="panel-title">
                    <span>🛠️</span> Maintenance Recommendations
                </span>
                {data && <StatusBadge status={data.status} />}
            </div>

            {loading && !data && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {[...Array(3)].map((_, i) => <div className="skeleton" key={i} style={{ height: 80, borderRadius: 12 }} />)}
                </div>
            )}

            {error && !data && (
                <div className="state-message">
                    <span className="state-message-icon">⚠️</span>
                    <span className="state-message-title">Could not load recommendations</span>
                </div>
            )}

            {data && recs.length === 0 && (
                <div className="state-message">
                    <span className="state-message-icon">✅</span>
                    <span className="state-message-title">{data.note ?? 'No active recommendations'}</span>
                    <span className="state-message-sub">
                        {data.status === 'not_trained'
                            ? 'Train the anomaly model first to generate recommendations.'
                            : 'All monitored signals are within normal bounds.'}
                    </span>
                </div>
            )}

            {recs.length > 0 && (
                <div className="rec-list">
                    {recs.map((rec, i) => (
                        <div className={`rec-card urgency-${rec.urgency}`} key={i}>
                            <div className="rec-card-header">
                                <div>
                                    <div className="rec-card-title">{rec.detected_issue}</div>
                                    <div className="rec-card-signal">
                                        Signal type: {rec.signal_type} · Risk: {rec.risk_level}
                                    </div>
                                </div>
                                <span className={`badge badge-${rec.urgency === 'high' ? 'danger' : rec.urgency === 'medium' ? 'warning' : 'success'}`}>
                                    {rec.urgency}
                                </span>
                            </div>
                            <div className="rec-inspect">
                                <strong>Suggested inspection:</strong>
                                <ul>
                                    {rec.suggested_inspection.map((item, j) => (
                                        <li key={j}>{item}</li>
                                    ))}
                                </ul>
                            </div>
                            <div style={{ marginTop: 'var(--space-2)', fontSize: '0.7rem', color: 'var(--clr-text-label)' }}>
                                Confidence: {rec.confidence} · {rec.limitations}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
