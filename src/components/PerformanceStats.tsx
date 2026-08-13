interface Props {
  isStreaming: boolean;
  status: string;
}

export function PerformanceStats({ isStreaming, status }: Props) {
  return (
    <div className="preview-header">
      <span>Status: <strong>{status}</strong></span>
      <span>FPS: <strong>{isStreaming ? 30 : 0}</strong></span>
    </div>
  );
}