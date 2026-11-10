interface BadgeProps {
    status: string
}

const map: Record<string, { cls: string; label: string }> = {
    ok: { cls: 'success', label: 'OK' },
    no_data: { cls: 'muted', label: 'No Data' },
    not_trained: { cls: 'muted', label: 'Not Trained' },
    insufficient_data: { cls: 'warning', label: 'Insufficient Data' },
    prediction_available: { cls: 'success', label: 'Available' },
    model_unavailable: { cls: 'danger', label: 'Unavailable' },
    anomalous: { cls: 'danger', label: 'Anomalous' },
    normal: { cls: 'success', label: 'Normal' },
    not_scored: { cls: 'muted', label: 'Not Scored' },
}

export default function StatusBadge({ status }: BadgeProps) {
    const m = map[status] ?? { cls: 'muted', label: status }
    return <span className={`badge badge-${m.cls}`}>{m.label}</span>
}
