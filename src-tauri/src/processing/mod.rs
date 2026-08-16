use anyhow::Result;
//use image::*;
use ndarray::Array4;

pub mod nms;
pub mod scrfd;
pub mod detector;
pub mod inswapper;  
pub mod transform;

pub fn preprocess_frame(
    raw_bytes: &[u8],
    width: u32,
    height: u32,
    target_width: usize,
    target_height: usize,
) -> Result<Array4<f32>> {
    let img_buf = image::RgbaImage::from_raw(width, height, raw_bytes.to_vec())
        .ok_or_else(|| anyhow::anyhow!("Failed to construct RgbaImage from bytes"))?;

    let resized = image::imageops::resize(
        &img_buf,
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::Triangle,
    );

    let mut tensor = Array4::<f32>::zeros((1, 3, target_height, target_width));

    for y in 0..target_height {
        for x in 0..target_width {
            let pixel = resized.get_pixel(x as u32, y as u32);
            tensor[[0, 0, y, x]] = (pixel[0] as f32) / 255.0; // R
            tensor[[0, 1, y, x]] = (pixel[1] as f32) / 255.0; // G
            tensor[[0, 2, y, x]] = (pixel[2] as f32) / 255.0; // B
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_frame_dimensions_and_normalization() {
        let width = 640;
        let height = 480;
        let fake_rgba_data = vec![255u8; (width * height * 4) as usize];

        let target_w = 640;
        let target_h = 640;

        let tensor = preprocess_frame(&fake_rgba_data, width, height, target_w, target_h)
            .expect("Preprocessing failed");

        assert_eq!(tensor.shape(), &[1, 3, target_h, target_w]);

        let sample_pixel_r = tensor[[0, 0, 0, 0]];
        assert!((sample_pixel_r - 1.0).abs() < f32::EPSILON);
    }
}