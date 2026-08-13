import { CameraDeviceInfo } from "../hooks/useCamera";

interface Props {
  cameras: CameraDeviceInfo[];
  selected: number | null;
  onSelect: (index: number) => void;
  onRefresh: () => void;
  disabled: boolean;
}

export function CameraSelector({ cameras, selected, onSelect, onRefresh, disabled }: Props) {
  return (
    <div className="card">
      <h2>Camera Settings</h2>
      <div className="control-group">
        <label htmlFor="camera-select">Device:</label>
        <select
          id="camera-select"
          value={selected ?? ""}
          onChange={(e) => onSelect(Number((e.target as HTMLSelectElement).value))}
          disabled={disabled}
        >
          {cameras.map((cam) => (
            <option key={cam.index} value={cam.index}>
              [{cam.index}] {cam.name}
            </option>
          ))}
        </select>
        <button onClick={onRefresh} disabled={disabled}>
          Refresh
        </button>
      </div>
    </div>
  );
}