# Saccade

## Overview

**Saccade** is a high-performance desktop application designed for real-time video stream manipulation and face swapping. Built with a systems-first approach, it decouples the native GUI and processing pipeline from the heavy machine-learning inference engine.

By leveraging **Rust** for frame orchestration, concurrency, and zero-copy memory management, alongside **ONNX Runtime** for hardware-accelerated model execution, Saccade achieves ultra-low latency suitable for live streaming and local video processing without Python runtime overhead.

---

## Architecture & Tech Stack

* **Core & Pipeline:** Rust (Async Tokio channels, thread pooling, zero-copy buffer management)
* **Desktop Shell:** Tauri v2 (Native OS window management, IPC bridge)
* **Frontend UI:** Preact + Vite (Lightweight, reactive control panel)
* **Inference Engine:** ONNX Runtime (`ort` crate) with hardware acceleration support (CUDA / DirectML / CoreML)
* **Computer Vision:** OpenCV bindings / custom image processing shaders for face cropping, alignment, and blending

---

## Project Structure

```text
Saccade/
├── src-tauri/             # Rust backend core
│   ├── src/
│   │   ├── models/        # Directory for ONNX weight files
│   │   ├── pipeline/      # Frame capture, queue management, and threading
│   │   ├── inference/     # ONNX Runtime wrappers (Detector & Inswapper)
│   │   ├── processing/    # Face alignment, color correction, and blending
│   │   └── main.rs        # Tauri application entry point
│   └── Cargo.toml
├── src/                   # Frontend UI (Preact)
└── README.md

```

---

## Development Roadmap

<details open>
<summary><b>Phase 1: Core Architecture & Pipeline Foundation</b></summary>

* [x] Initialize Tauri v2 workspace and Rust backend structure.
* [x] Implement async frame capture pipeline for local webcams and video files.
* [x] Set up zero-copy frame buffer pooling to minimize memory allocations.
* [x] Integrate basic Preact UI dashboard for device selection and previews.

</details>

<details open>
<summary><b>Phase 2: Inference Integration</b></summary>

* [x] Configure ONNX Runtime (`ort`) with CUDA and CPU execution providers.
* [x] Optimize tensor conversion pipelines between image buffers and model inputs.
* [x] Implement post-processing core utilities (NMS algorithm, IoU math, and detection data types).
* [ ] Implement SCRFD / RetinaFace model loader & anchor decoder for high-speed face detection and landmark extraction.
* [ ] Integrate InsightFace (`inswapper_128.onnx`) embedding extraction and face-swapping inference loop.

</details>

<details>
<summary><b>Phase 3: Post-Processing & Blending Optimization</b></summary>

* [ ] Implement facial alignment and affine transformation matrices.
* [ ] Add seamless blending algorithms (Poisson blending / Mask feathering) to eliminate harsh edges.
* [ ] Implement temporal smoothing filters to reduce jitter and flickering across consecutive video frames.

</details>

<details>
<summary><b>Phase 4: Advanced Vision & Virtual Camera Pipeline</b></summary>

* [ ] Integrate native Virtual Camera driver support (OBS VirtualCam / v4l2loopback IPC).
* [ ] Implement multi-face tracking and target selection (ID-based face locking).
* [ ] Add real-time facial expression transfer / reenactment module (e.g., LivePortrait / FOMM integration).
* [ ] Support custom target mask fine-tuning and Occlusion Aware Blending (hair, hands, glasses handling).

</details>

<details>
<summary><b>Phase 5: Audio, Latency Optimization & Security Controls</b></summary>

* [ ] Add real-time Voice Conversion (RVC) pipeline synchronized with video output streams.
* [ ] Implement hardware-accelerated NVENC / VAAPI / AMF video encoding for low-bitrate streaming.
* [ ] Integrate local model weight encryption and secure memory handling for proprietary weights.
* [ ] Add automated bench-harness for FPS, latency jitter, and VRAM consumption profiling.

</details>

<details>
<summary><b>Phase 6: Performance Tuning & Release</b></summary>

* [ ] Benchmark end-to-end latency and throughput across various GPU architectures.
* [ ] Implement multi-threaded frame dropping and queue backpressure handling for real-time streaming.
* [ ] Package production binaries with bundled ONNX runtimes and asset management.

</details>

---

## Getting Started

### Prerequisites

* Rust toolchain (stable)
* Node.js & pnpm / npm
* CUDA Toolkit (optional, recommended for GPU acceleration)

### Building from Source

```bash
git clone https://github.com/itsventie/Saccade.git
cd Saccade
pnpm install
pnpm tauri dev
```