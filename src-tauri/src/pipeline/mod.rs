use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use anyhow::{anyhow, Result};
use bytes::Bytes;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array4, ArrayView3};
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use serde::Serialize;
use tracing::{error, info};

use crate::inference::types::DetectedFace;
use crate::processing::detector::ScrfdDetector;
use crate::processing::inswapper::InswapperPipeline;

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

impl VideoFrame {
    pub fn to_input_tensor(&self) -> Result<Array4<f32>> {
        let (h, w) = (self.height as usize, self.width as usize);
        if self.data.len() != w * h * 4 {
            return Err(anyhow!(
                "Invalid frame buffer length: expected {}, got {}",
                w * h * 4,
                self.data.len()
            ));
        }

        // RGBA [H, W, 4]
        let rgba_view = ArrayView3::from_shape((h, w, 4), &self.data)
            .map_err(|e| anyhow!("Failed to view frame data: {e}"))?;

        let mut tensor = Array4::<f32>::zeros((1, 3, h, w));

        for y in 0..h {
            for x in 0..w {
                let r = rgba_view[[y, x, 0]] as f32;
                let g = rgba_view[[y, x, 1]] as f32;
                let b = rgba_view[[y, x, 2]] as f32;

                tensor[[0, 0, y, x]] = r;
                tensor[[0, 1, y, x]] = g;
                tensor[[0, 2, y, x]] = b;
            }
        }

        Ok(tensor)
    }
}

pub struct FramePipeline {
    #[allow(dead_code)]
    is_running: Arc<AtomicBool>,
    pub frame_rx: Receiver<VideoFrame>,
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

pub struct FaceSwapPipeline {
    detector: ScrfdDetector,
    swapper: InswapperPipeline,
    source_embedding: Option<Vec<f32>>,
}

impl FaceSwapPipeline {
    pub fn new(scrfd_model_path: &str, inswapper_model_path: &str) -> Result<Self> {
        let detector = ScrfdDetector::new(scrfd_model_path, 640, 640, 0.4, 0.5)?;
        let swapper = InswapperPipeline::new(inswapper_model_path)?;

        Ok(Self {
            detector,
            swapper,
            source_embedding: None,
        })
    }

    pub fn set_source_embedding(&mut self, embedding: Vec<f32>) -> Result<()> {
        InswapperPipeline::validate_embedding(&embedding)?;
        self.source_embedding = Some(embedding);
        Ok(())
    }

    pub fn process_video_frame(
        &mut self,
        frame: &VideoFrame,
    ) -> Result<(Vec<DetectedFace>, Option<Array4<f32>>)> {
        let input_tensor = frame.to_input_tensor()?;
        let detected_faces = self.detector.detect(input_tensor)?;

        if detected_faces.is_empty() {
            return Ok((vec![], None));
        }

        let swapped_crop = if let Some(ref embedding) = self.source_embedding {
            let target_crop = Array4::zeros((1, 3, 128, 128));
            let swapped = self.swapper.swap_face(target_crop, embedding)?;
            Some(swapped)
        } else {
            None
        };

        Ok((detected_faces, swapped_crop))
    }
}