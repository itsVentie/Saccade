import { useCamera } from "./hooks/useCamera";
import { Header } from "./components/Header";
import { CameraSelector } from "./components/CameraSelector";
import { ControlPanel } from "./components/ControlPanel";
import { PerformanceStats } from "./components/PerformanceStats";
import { Viewport } from "./components/Viewport";
import "./App.css";

export function App() {
  const {
    cameras,
    selectedCamera,
    selectCamera,
    isStreaming,
    toggleStream,
    refreshCameras,
    status,
  } = useCamera();

  return (
    <div className="container">
      <Header />
      <div className="main-content">
        <aside className="sidebar">
          <CameraSelector
            cameras={cameras}
            selected={selectedCamera}
            onSelect={selectCamera}
            onRefresh={refreshCameras}
            disabled={isStreaming}
          />
          <ControlPanel
            isStreaming={isStreaming}
            onToggleStream={toggleStream}
            disabled={selectedCamera === null}
          />
        </aside>
        <section className="card preview-card">
          <PerformanceStats isStreaming={isStreaming} status={status} />
          <Viewport isStreaming={isStreaming} selectedCamera={selectedCamera} />
        </section>
      </div>
    </div>
  );
}

export default App;