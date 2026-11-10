import type { RULResponse } from '../api'
import StatusBadge from './StatusBadge'

interface Props {
    data: RULResponse | null
    loading: boolean
    error: boolean
}

function ArcGauge({ value, max, label }: { value: number | null; max: number; label: string }) {
    const r = 70
    const cx = 90
    const cy = 90
    const startAngle = -210
    const endAngle = 30
    const totalDeg = endAngle - startAngle  // 240°
    const pct = value !== null ? Math.min(1, value / max) : 0
    const deg = startAngle + totalDeg * pct

    function polar(ang: number, radius: number) {
        const rad = (ang * Math.PI) / 180
        return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) }
    }

    function arc(a1: number, a2: number, radius: number) {
        const p1 = polar(a1, radius)
        const p2 = polar(a2, radius)
        const large = a2 - a1 > 180 ? 1 : 0
        return `M ${p1.x} ${p1.y} A ${radius} ${radius} 0 ${large} 1 ${p2.x} ${p2.y}`
    }

    const needle = polar(deg, r - 8)
    const needleBase = polar(deg, 8)

    const colour = value !== null
        ? value < max * 0.25 ? 'var(--clr-danger)'
            : value < max * 0.5 ? 'var(--clr-warning)'
                : 'var(--clr-success)'
        : 'var(--clr-muted)'

    return (
        <svg viewBox="0 0 180 120" width={180} height={120} className="gauge-svg">
            {/* Track */}
            <path d={arc(startAngle, endAngle, r)} fill="none" stroke="hsla(230,30%,30%,0.5)" strokeWidth={12} strokeLinecap="round" />
            {/* Fill */}
            {value !== null && (
                <path d={arc(startAngle, deg, r)} fill="none" stroke={colour} strokeWidth={12} strokeLinecap="round" />
            )}
            {/* Needle */}
            <line x1={needleBase.x} y1={needleBase.y} x2={needle.x} y2={needle.y}
                stroke={colour} strokeWidth={3} strokeLinecap="round" />
            <circle cx={cx} cy={cy} r={6} fill={colour} />
            {/* Label */}
            <text x={cx} y={cy + 28} textAnchor="middle" fill="var(--clr-text-dim)" fontSize={11}>{label}</text>
        </svg>
    )
}

export default function RULPanel({ data, loading, error }: Props) {
    return (
        <div className="panel col-4">
            <div className="panel-header">
                <span className="panel-title">
                    <span>⏱️</span> Remaining Useful Life
                </span>
                {data && <StatusBadge status={data.status} />}
            </div>

            {loading && !data && <div className="skeleton" style={{ height: 120, borderRadius: 12 }} />}

            {error && !data && (
                <div className="state-message">
                    <span className="state-message-icon">⚠️</span>
                    <span className="state-message-title">Could not load RUL</span>
                </div>
            )}

            {data && data.rul_hours === null && (
                <div className="state-message">
                    <span className="state-message-icon">🔬</span>
                    <span className="state-message-title">
                        {data.status === 'not_trained' ? 'RUL model not trained' : 'Unavailable'}
                    </span>
                    <span className="state-message-sub">{data.message}</span>
                </div>
            )}

            {data?.rul_hours !== null && data?.rul_hours !== undefined && (
                <div className="gauge-wrap">
                    <ArcGauge value={data.rul_hours} max={2000} label="h remaining" />
                    <div>
                        <div className="metric-value" style={{ textAlign: 'center' }}>
                            {data.rul_hours.toFixed(0)}<span style={{ fontSize: '1.2rem', fontWeight: 400 }}> h</span>
                        </div>
                        <div className="metric-label" style={{ textAlign: 'center' }}>Estimated remaining operating hours</div>
                    </div>
                </div>
            )}
        </div>
    )
}
