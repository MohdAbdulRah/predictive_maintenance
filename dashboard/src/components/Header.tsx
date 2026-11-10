interface Props {
    connected: boolean
    lastUpdated: Date | null
    machineId: string | null
}

export default function Header({ connected, lastUpdated, machineId }: Props) {
    return (
        <header className="header">
            <div className="header-left">
                <div className="header-logo" aria-hidden="true">⚡</div>
                <div>
                    <div className="header-title">Predictive Maintenance</div>
                    <div className="header-sub">MTConnect · {machineId ?? 'no device'}</div>
                </div>
            </div>
            <div className="header-right">
                <div className="connection-indicator">
                    <span className={`dot ${connected ? 'dot-success' : 'dot-danger'}`} />
                    {connected ? 'Live' : 'Disconnected'}
                </div>
                {lastUpdated && (
                    <div className="last-updated">
                        Updated {lastUpdated.toLocaleTimeString()}
                    </div>
                )}
            </div>
        </header>
    )
}
