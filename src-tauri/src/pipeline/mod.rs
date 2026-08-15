use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use anyhow::Result;
use bytes::Bytes;
use crossbeam_channel::{bounded, Receiver, Sender};
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use serde::Serialize;
use tracing::{error, info};

#[derive(Debug, Clone, Serialize)]
pub struct CameraDeviceInfo {
    pub index: u32,
    pub name: String,
}

pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub data: Bytes,
    pub timestamp_ms: u64,
}

pub struct FramePipeline {
    #[allow(dead_code)]
    is_running: Arc<AtomicBool>,
    #[allow(dead_code)]
    frame_rx: Receiver<VideoFrame>,
}

impl FramePipeline {
    pub fn list_cameras() -> Result<Vec<CameraDeviceInfo>> {
        let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Auto)?;
        let mut devices = Vec::new();

        for cam in cameras {
            devices.push(CameraDeviceInfo {
                index: match cam.index() {
                    CameraIndex::Index(idx) => *idx,
                    _ => 0,
                },
                name: cam.human_name(),
            });
        }

        Ok(devices)
    }

    pub fn start_capture(camera_idx: u32, queue_capacity: usize) -> Result<(Self, Sender<()>)> {
        let (frame_tx, frame_rx) = bounded::<VideoFrame>(queue_capacity);
        let (stop_tx, stop_rx) = bounded::<()>(1);
        let is_running = Arc::new(AtomicBool::new(true));
        let is_running_thread = is_running.clone();

        std::thread::spawn(move || {
            let index = CameraIndex::Index(camera_idx);
            let requested = RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

            let mut camera = match Camera::new(index, requested) {
                Ok(cam) => cam,
                Err(err) => {
                    error!("Failed to open camera device: {err}");
                    return;
                }
            };

            if let Err(err) = camera.open_stream() {
                error!("Failed to open camera stream: {err}");
                return;
            }

            info!("Camera stream started successfully");

            while is_running_thread.load(Ordering::Relaxed) {
                if stop_rx.try_recv().is_ok() {
                    break;
                }

                match camera.frame() {
                    Ok(frame) => {
                        let decoded = match frame.decode_image::<RgbAFormat>() {
                            Ok(img) => img,
                            Err(err) => {
                                error!("Frame decode error: {err}");
                                continue;
                            }
                        };

                        let video_frame = VideoFrame {
                            width: decoded.width(),
                            height: decoded.height(),
                            data: Bytes::from(decoded.into_raw()),
                            timestamp_ms: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64,
                        };
                        
                        let _ = frame_tx.try_send(video_frame);
                    }
                    Err(err) => {
                        error!("Failed to capture frame: {err}");
                    }
                }
            }

            let _ = camera.stop_stream();
            info!("Camera stream stopped");
        });

        Ok((Self { is_running, frame_rx }, stop_tx))
    }
}