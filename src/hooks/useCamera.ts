import { useEffect, useState } from "preact/hooks";
import { invoke } from "@tauri-apps/api/core";

export interface CameraDeviceInfo {
  index: number;
  name: string;
}

export function useCamera() {
  const [cameras, setCameras] = useState<CameraDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("Idle");

  const refreshCameras = async () => {
    try {
      setStatus("Scanning...");
      const devices = await invoke<CameraDeviceInfo[]>("get_available_cameras");
      setCameras(devices);
      if (devices.length > 0 && selectedCamera === null) {
        setSelectedCamera(devices[0].index);
      }
      setStatus("Ready");
    } catch (err) {
      console.error("Failed to list cameras:", err);
      setStatus(`Error: ${err}`);
    }
  };

  useEffect(() => {
    refreshCameras();
  }, []);

  const toggleStream = () => {
    setIsStreaming((prev) => !prev);
    setStatus(isStreaming ? "Stopped" : "Streaming...");
  };

  return {
    cameras,
    selectedCamera,
    selectCamera: setSelectedCamera,
    isStreaming,
    toggleStream,
    refreshCameras,
    status,
  };
}