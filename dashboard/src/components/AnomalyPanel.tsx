import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { AnomalyResponse } from '../api'
import StatusBadge from './StatusBadge'

interface Props {
    data: AnomalyResponse | null
    loading: boolean
    error: boolean
}

function scoreColor(score: number | null): string {
    if (score === null) return 'var(--clr-muted)'
    if (score > 0.85) return 'var(--clr-danger)'
    if (score > 0.6) return 'var(--clr-warning)'
    return 'var(--clr-success)'
}

export default function AnomalyPanel({ data, loading, error }: Props) {
    const results = data?.results ?? []
    const latest = results.length > 0 ? results[results.length - 1] : null
    const chartData = results
        .filter(r => r.anomaly_score !== null)
        .map(r => ({
            t: new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            score: r.anomaly_score !== null ? +(r.anomaly_score * 100).toFixed(1) : null,
            st: r.status,
        }))

    const topFeatures = latest?.top_features ?? []

    return (
        <div className="panel col-6">
            <div className="panel-header">
                <span className="panel-title">
                    <span>🔍</span> Anomaly Detection
                </span>
                {data && <StatusBadge status={data.status} />}
            </div>

            {/* State gates */}
            {!loading && (data?.status === 'not_trained' || data?.status === 'insufficient_data') && (
                <div className="state-message">
                    <span className="state-message-icon">🤖</span>
                    <span className="state-message-title">{data.status === 'not_trained' ? 'Model not trained' : 'Insufficient data'}</span>
                    <span className="state-message-sub">{data.message}</span>
                    {data.status === 'not_trained' && (
                        <code style={{ fontSize: '0.72rem', color: 'var(--clr-accent)', fontFamily: 'var(--font-mono)', marginTop: 8 }}>
                            python -m training.train_anomaly
                        </code>
                    )}
                </div>
            )}

            {loading && !data && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div className="skeleton" style={{ height: 180 }} />
                    <div className="skeleton" style={{ width: '60%' }} />
                </div>
            )}

            {error && !data && (
                <div className="state-message">
                    <span className="state-message-icon">⚠️</span>
                    <span className="state-message-title">Could not load anomaly data</span>
                </div>
            )}

            {data?.status === 'prediction_available' && results.length > 0 && (
                <>
                    {/* Big score */}
                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 'var(--space-4)', marginBottom: 'var(--space-5)' }}>
                        <div>
                            <div
                                className="metric-value"
                                style={{ background: 'none', WebkitTextFillColor: latest ? scoreColor(latest.anomaly_score) : 'var(--clr-muted)' }}
                            >
                                {latest?.anomaly_score !== null && latest?.anomaly_score !== undefined
                                    ? `${(latest.anomaly_score * 100).toFixed(0)}%`
                                    : '—'}
                            </div>
                            <div className="metric-label">Anomaly percentile (latest)</div>
                        </div>
                        {latest && <StatusBadge status={latest.status} />}
                        {latest && (
                            <span style={{ fontSize: '0.78rem', color: 'var(--clr-text-dim)' }}>
                                State: <strong>{latest.operating_state}</strong>
                            </span>
                        )}
                    </div>

                    {/* Chart */}
                    {chartData.length > 1 && (
                        <div className="timeline-wrap">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="aGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="hsl(252, 94%, 67%)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="hsl(252, 94%, 67%)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="hsla(230,40%,40%,0.12)" />
                                    <XAxis dataKey="t" tick={{ fill: 'var(--clr-text-label)', fontSize: 10 }} tickLine={false} axisLine={false} />
                                    <YAxis domain={[0, 100]} tick={{ fill: 'var(--clr-text-label)', fontSize: 10 }} tickLine={false} axisLine={false} unit="%" />
                                    <Tooltip
                                        contentStyle={{ background: 'var(--clr-bg-card)', border: '1px solid var(--clr-border)', borderRadius: 8, fontSize: 12 }}
                                        labelStyle={{ color: 'var(--clr-text-dim)' }}
                                        itemStyle={{ color: 'var(--clr-accent)' }}
                                    />
                                    <Area type="monotone" dataKey="score" name="Score" stroke="hsl(252,94%,67%)" strokeWidth={2} fill="url(#aGrad)" dot={false} connectNulls={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Top features */}
                    {topFeatures.length > 0 && (
                        <>
                            <div className="divider" />
                            <div style={{ fontSize: '0.75rem', color: 'var(--clr-text-label)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                Evidence features
                            </div>
                            <div className="features-list">
                                {topFeatures.slice(0, 8).map((f, i) => (
                                    <span className="feature-chip" key={i} title={`z = ${f.z?.toFixed(2)}`}>
                                        {f.feature}{f.z ? ` (z=${f.z.toFixed(1)})` : ''}
                                    </span>
                                ))}
                            </div>
                        </>
                    )}

                    {/* Disclaimer */}
                    {data.disclaimer && <div className="disclaimer">⚠️ {data.disclaimer}</div>}
                </>
            )}
        </div>
    )
}
