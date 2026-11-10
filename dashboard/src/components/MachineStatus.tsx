import type { LiveResponse } from '../api'
import StatusBadge from './StatusBadge'

interface Props {
    data: LiveResponse | null
    loading: boolean
    error: boolean
}

const categoryIcon: Record<string, string> = {
    SAMPLE: '📊', EVENT: '🔔', CONDITION: '⚠️',
}

function fmtValue(v: number | string | null, unavailable: boolean, units: string | null): string {
    if (unavailable) return 'UNAVAILABLE'
    if (v === null || v === undefined) return '—'
    if (typeof v === 'number') return `${v.toFixed(3)}${units ? ' ' + units : ''}`
    return String(v)
}

export default function MachineStatus({ data, loading, error }: Props) {
    const obs = data?.observations ?? []
    const samples = obs.filter(o => o.category === 'SAMPLE')
    const events = obs.filter(o => o.category === 'EVENT')
    const conds = obs.filter(o => o.category === 'CONDITION')

    return (
        <div className="panel col-12">
            <div className="panel-header">
                <span className="panel-title">
                    <span>{categoryIcon.SAMPLE}</span> Machine Live Status
                </span>
                {data && <StatusBadge status={data.status} />}
            </div>

            {loading && !data && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {[...Array(5)].map((_, i) => <div className="skeleton" key={i} style={{ height: 28 }} />)}
                </div>
            )}

            {error && !data && (
                <div className="state-message">
                    <span className="state-message-icon">🔌</span>
                    <span className="state-message-title">Cannot reach API server</span>
                    <span className="state-message-sub">Make sure the backend is running on port 8000</span>
                </div>
            )}

            {data?.status === 'no_data' && (
                <div className="state-message">
                    <span className="state-message-icon">📡</span>
                    <span className="state-message-title">No telemetry collected yet</span>
                    <span className="state-message-sub">
                        Run <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--clr-accent)' }}>
                            python -m scripts.collect
                        </code> to start collecting from the MTConnect agent.
                    </span>
                </div>
            )}

            {data?.status === 'ok' && obs.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-5)' }}>
                    {/* SAMPLE signals */}
                    <div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--clr-text-label)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 'var(--space-3)', fontWeight: 600 }}>
                            {categoryIcon.SAMPLE} Sensor Readings ({samples.length})
                        </div>
                        <div className="obs-scroll">
                            <table className="obs-table">
                                <thead>
                                    <tr>
                                        <th>Signal</th>
                                        <th>Type</th>
                                        <th>Value</th>
                                        <th>Updated</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {samples.map(o => (
                                        <tr key={o.data_item_id}>
                                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{o.name || o.data_item_id}</td>
                                            <td style={{ color: 'var(--clr-text-label)', fontSize: '0.72rem' }}>{o.type}</td>
                                            <td className={o.unavailable ? 'obs-na' : 'obs-val'}>
                                                {fmtValue(o.value as number | string | null, o.unavailable, o.units)}
                                            </td>
                                            <td style={{ color: 'var(--clr-text-label)', fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                                                {new Date(o.timestamp).toLocaleTimeString()}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Events + Conditions */}
                    <div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--clr-text-label)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 'var(--space-3)', fontWeight: 600 }}>
                            {categoryIcon.EVENT} Events &amp; Conditions ({events.length + conds.length})
                        </div>
                        <div className="obs-scroll">
                            <table className="obs-table">
                                <thead>
                                    <tr>
                                        <th>Signal</th>
                                        <th>Cat.</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {[...events, ...conds].map(o => (
                                        <tr key={o.data_item_id}>
                                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{o.name || o.data_item_id}</td>
                                            <td style={{ color: 'var(--clr-text-label)', fontSize: '0.72rem' }}>
                                                {categoryIcon[o.category] ?? ''} {o.category}
                                            </td>
                                            <td className={o.unavailable ? 'obs-na' : 'obs-val'}>
                                                {o.condition_level
                                                    ? <span style={{ color: o.condition_level === 'FAULT' ? 'var(--clr-danger)' : o.condition_level === 'WARNING' ? 'var(--clr-warning)' : 'var(--clr-success)' }}>{o.condition_level}</span>
                                                    : fmtValue(o.value as number | string | null, o.unavailable, o.units)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {data && data.status === 'ok' && (
                <div style={{ marginTop: 'var(--space-3)', fontSize: '0.7rem', color: 'var(--clr-text-label)', fontFamily: 'var(--font-mono)' }}>
                    machine_id: {data.machine_id} · generated {new Date(data.generated_at).toLocaleTimeString()}
                </div>
            )}
        </div>
    )
}
