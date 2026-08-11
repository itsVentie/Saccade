// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod inference;
pub mod pipeline;
pub mod processing;

use pipeline::{CameraDeviceInfo, FramePipeline};

#[tauri::command]
fn get_available_cameras() -> Result<Vec<CameraDeviceInfo>, String> {
    FramePipeline::list_cameras().map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
