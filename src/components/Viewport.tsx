interface Props {
  isStreaming: boolean;
  selectedCamera: number | null;
}

export function Viewport({ isStreaming, selectedCamera }: Props) {
  return (
    <div className="viewport-placeholder">
      {isStreaming ? (
        <div className="active-stream">
          <p>Stream active for device #{selectedCamera}</p>
        </div>
      ) : (
        <p className="idle-text">Camera stream is offline</p>
      )}
    </div>
  );
}