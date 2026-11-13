/* CSV Prediction Page - Upload CSV, get AI4I machine failure predictions */

import { useCallback, useRef, useState } from 'react'

const API = (import.meta as any).env?.VITE_API ?? ''

// ---- Types ----

interface ModelStatus {
    trained: boolean
    best_model?: string
    threshold?: number
    feature_columns?: string[]
    test_metrics?: Record<string, number | null>
    all_models?: string[]
    message?: string
}

interface PredictionRow {
    Type?: string
    'Air temperature'?: number
    'Process temperature'?: number
    'Rotational speed'?: number
    'Torque'?: number
    'Tool wear'?: number
    failure_probability: number | null
    predicted_failure: 0 | 1
    predicted_label: string
    [key: string]: unknown
}

interface PredictionResult {
    model_used: string
    threshold: number
    total_rows: number
    predicted_failures: number
    failure_rate: number
    results: PredictionRow[]
}

// ---- Component ----

export default function CsvPredictPage() {
    const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
    const [loadingStatus, setLoadingStatus] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [uploading, setUploading] = useState(false)
    const [dragOver, setDragOver] = useState(false)
    const [fileName, setFileName] = useState<string | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const checkStatus = useCallback(async () => {
        setLoadingStatus(true)
        try {
            const res = await fetch(`${API}/api/csv-predict/status`)
            const data = await res.json()
            setModelStatus(data)
        } catch (e) {
            setError('Failed to reach API. Is the backend running?')
        } finally {
            setLoadingStatus(false)
        }
    }, [])

    const uploadFile = useCallback(async (file: File) => {
        setUploading(true)
        setError(null)
        setResult(null)
        setFileName(file.name)

        const formData = new FormData()
        formData.append('file', file)

        try {
            const res = await fetch(`${API}/api/csv-predict/predict`, {
                method: 'POST',
                body: formData,
            })
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}))
                throw new Error(errData.detail || `HTTP ${res.status}`)
            }
            const data = await res.json()
            setResult(data)
        } catch (e: any) {
            setError(e.message || 'Unknown error')
        } finally {
            setUploading(false)
        }
    }, [])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) uploadFile(file)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setDragOver(false)
        const file = e.dataTransfer.files?.[0]
        if (file) uploadFile(file)
    }

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        setDragOver(true)
    }

    const handleDragLeave = () => setDragOver(false)

    const featureCols = ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']

    return (
        <div className="csv-predict-page">
            <div className="csv-header">
                <h2>🏭 AI4I Machine Failure Predictor</h2>
                <p className="csv-subtitle">
                    Upload a CSV file with machine sensor readings to predict failure probability using all 6 trained models.
                </p>
                <button className="btn-status" onClick={checkStatus} disabled={loadingStatus}>
                    {loadingStatus ? 'Checking…' : '🔍 Check Model Status'}
                </button>
            </div>

            {modelStatus && (
                <div className={`model-status-box ${modelStatus.trained ? 'trained' : 'not-trained'}`}>
                    {modelStatus.trained ? (
                        <>
                            <div className="status-row">
                                <span className="status-badge green">✅ Models Trained</span>
                                <span>Best: <strong>{modelStatus.best_model}</strong></span>
                                <span>Threshold: <strong>{modelStatus.threshold}</strong></span>
                            </div>
                            {modelStatus.test_metrics && (
                                <div className="metrics-row">
                                    {Object.entries(modelStatus.test_metrics).map(([k, v]) =>
                                        v != null ? (
                                            <span key={k} className="metric-chip">
                                                {k.replace(/_/g, ' ')}: <strong>{(v * 100).toFixed(1)}%</strong>
                                            </span>
                                        ) : null
                                    )}
                                </div>
                            )}
                            {modelStatus.all_models && (
                                <div className="models-list">
                                    Available models: {modelStatus.all_models.join(', ')}
                                </div>
                            )}
                        </>
                    ) : (
                        <div>
                            <span className="status-badge red">❌ Not Trained</span>
                            <p>{modelStatus.message}</p>
                            <code>python -m training.train_notebook_models</code>
                        </div>
                    )}
                </div>
            )}

            <div className="csv-format-hint">
                <strong>Required CSV columns:</strong>{' '}
                {featureCols.map(c => <code key={c}>{c}</code>)}
                <br />
                <span className="hint-note">
                    Units like <code>[K]</code>, <code>[rpm]</code>, <code>[Nm]</code>, <code>[min]</code> are stripped automatically.
                    Extra columns (UDI, Product ID, etc.) are ignored.
                </span>
            </div>

            <div
                className={`drop-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                />
                {uploading ? (
                    <div className="uploading-spinner">
                        <div className="spinner" />
                        <span>Analyzing {fileName}…</span>
                    </div>
                ) : (
                    <>
                        <div className="drop-icon">📂</div>
                        <div className="drop-text">
                            {fileName ? `✓ ${fileName}` : 'Drop CSV here or click to browse'}
                        </div>
                        <div className="drop-hint">Supports AI4I 2020 format</div>
                    </>
                )}
            </div>

            {error && (
                <div className="error-box">
                    <strong>⚠️ Error:</strong> {error}
                </div>
            )}

            {result && (
                <div className="results-section">
                    <div className="results-summary">
                        <div className="summary-card">
                            <div className="summary-num">{result.total_rows}</div>
                            <div className="summary-label">Total rows</div>
                        </div>
                        <div className="summary-card failure">
                            <div className="summary-num">{result.predicted_failures}</div>
                            <div className="summary-label">Predicted failures</div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-num">{(result.failure_rate * 100).toFixed(1)}%</div>
                            <div className="summary-label">Failure rate</div>
                        </div>
                        <div className="summary-card model-card">
                            <div className="summary-label">Model used</div>
                            <div className="summary-model">{result.model_used}</div>
                            <div className="summary-label">threshold: {result.threshold}</div>
                        </div>
                    </div>

                    <div className="results-table-wrap">
                        <table className="results-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Type</th>
                                    <th>Air Temp</th>
                                    <th>Proc Temp</th>
                                    <th>Rot Speed</th>
                                    <th>Torque</th>
                                    <th>Tool Wear</th>
                                    <th>Failure Prob</th>
                                    <th>Prediction</th>
                                </tr>
                            </thead>
                            <tbody>
                                {result.results.map((row, i) => (
                                    <tr key={i} className={row.predicted_failure === 1 ? 'row-failure' : ''}>
                                        <td>{i + 1}</td>
                                        <td>{String(row['Type'] ?? '—')}</td>
                                        <td>{row['Air temperature'] != null ? Number(row['Air temperature']).toFixed(1) : '—'}</td>
                                        <td>{row['Process temperature'] != null ? Number(row['Process temperature']).toFixed(1) : '—'}</td>
                                        <td>{row['Rotational speed'] != null ? Number(row['Rotational speed']).toFixed(0) : '—'}</td>
                                        <td>{row['Torque'] != null ? Number(row['Torque']).toFixed(1) : '—'}</td>
                                        <td>{row['Tool wear'] != null ? Number(row['Tool wear']).toFixed(0) : '—'}</td>
                                        <td>
                                            {row.failure_probability != null
                                                ? <span className={`prob-badge ${row.failure_probability > 0.5 ? 'prob-high' : row.failure_probability > 0.2 ? 'prob-med' : 'prob-low'}`}>
                                                    {(row.failure_probability * 100).toFixed(1)}%
                                                  </span>
                                                : '—'}
                                        </td>
                                        <td className={row.predicted_failure === 1 ? 'pred-failure' : 'pred-ok'}>
                                            {row.predicted_label}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}

