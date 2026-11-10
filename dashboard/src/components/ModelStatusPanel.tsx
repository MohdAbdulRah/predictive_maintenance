import type { ModelStatusResponse } from '../api'

interface Props {
    data: ModelStatusResponse | null
    loading: boolean
    error: boolean
}

const stageLabels: Record<string, string> = {
    anomaly: 'Stage A — Anomaly',
    failure: 'Stage B — Failure',
    rul: 'Stage C — RUL',
}

const statusColor: Record<string, string> = {
    not_trained: 'var(--clr-muted)',
    insufficient_data: 'var(--clr-warning)',
    prediction_available: 'var(--clr-success)',
    model_unavailable: 'var(--clr-danger)',
    ok: 'var(--clr-success)',
}

export default function ModelStatusPanel({ data, loading, error }: Props) {
    return (
        <div className="panel col-12">
            <div className="panel-header">
                <span className="panel-title">
                    <span>⚙️</span> Model Registry &amp; Training Gates
                </span>
            </div>

            {loading && (
                <div className="model-stages">
                    {['anomaly', 'failure', 'rul'].map(s => (
                        <div className="model-stage-card" key={s}>
                            <div className="skeleton" style={{ width: '60%' }} />
                            <div className="skeleton" style={{ width: '40%', marginTop: 8 }} />
                        </div>
                    ))}
                </div>
            )}

            {error && !loading && (
                <div className="state-message">
                    <span className="state-message-icon">⚠️</span>
                    <span className="state-message-title">Could not load model status</span>
                </div>
            )}

            {data && !loading && (
                <div className="model-stages">
                    {Object.entries(data.stages).map(([key, stage]) => (
                        <div className="model-stage-card" key={key}>
                            <span className="model-stage-name">{stageLabels[key] ?? key}</span>
                            <span
                                className="model-stage-status"
                                style={{ color: statusColor[stage.status] ?? 'var(--clr-text-dim)' }}
                            >
                                {stage.status.replace(/_/g, ' ')}
                            </span>
                            <span className="model-stage-detail">
                                artifact: {stage.artifact_present ? '✓ present' : '✗ missing'}
                            </span>
                            {stage.trained_at && (
                                <span className="model-stage-detail">
                                    trained: {new Date(stage.trained_at).toLocaleString()}
                                </span>
                            )}
                            {!stage.artifact_present && key === 'anomaly' && (
                                <span className="model-stage-detail" style={{ color: 'var(--clr-warning)', marginTop: 4, fontSize: '0.68rem' }}>
                                    Run: python -m training.train_anomaly
                                </span>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
