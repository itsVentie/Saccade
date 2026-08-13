interface Props {
  isStreaming: boolean;
  onToggleStream: () => void;
  disabled: boolean;
}

export function ControlPanel({ isStreaming, onToggleStream, disabled }: Props) {
  return (
    <div className="card">
      <h2>Pipeline Controls</h2>
      <button
        className={isStreaming ? "btn-danger" : "btn-primary"}
        onClick={onToggleStream}
        disabled={disabled}
      >
        {isStreaming ? "Stop Stream" : "Start Stream"}
      </button>
    </div>
  );
}