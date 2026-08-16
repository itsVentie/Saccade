use ndarray::{Array2, Array4};
use super::transform::AffineMatrix2x3;

pub fn generate_feathered_mask(size: usize, margin_ratio: f32) -> Array2<f32> {
    let mut mask = Array2::<f32>::zeros((size, size));
    let center = (size as f32 - 1.0) / 2.0;
    let radius = center * (1.0 - margin_ratio.clamp(0.0, 0.8));

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= radius {
                mask[[y, x]] = 1.0;
            } else if dist < center {
                let factor = (dist - radius) / (center - radius);
                let alpha = 0.5 * (1.0 + (factor * std::f32::consts::PI).cos());
                mask[[y, x]] = alpha;
            } else {
                mask[[y, x]] = 0.0;
            }
        }
    }

    mask
}

pub fn warp_affine_back(
    original_frame: &mut Array4<f32>,
    swapped_face_128: &Array4<f32>,
    affine_matrix: &AffineMatrix2x3,
    mask_128: &Array2<f32>,
) {
    let inv_mat = match affine_matrix.invert() {
        Some(inv) => inv,
        None => return,
    };

    let h_src = original_frame.shape()[2] as f32;
    let w_src = original_frame.shape()[3] as f32;

    for y_dst in 0..128 {
        for x_dst in 0..128 {
            let alpha = mask_128[[y_dst, x_dst]];
            if alpha <= 1e-4 {
                continue;
            }

            let p_dst = [x_dst as f32, y_dst as f32];
            let p_src = inv_mat.transform_point(p_dst);

            let x_src = p_src[0];
            let y_src = p_src[1];

            if x_src >= 0.0 && x_src < w_src - 1.0 && y_src >= 0.0 && y_src < h_src - 1.0 {
                let x0 = x_src.floor() as usize;
                let y0 = y_src.floor() as usize;

                for c in 0..3 {
                    let orig_val = original_frame[[0, c, y0, x0]];
                    let swap_val = swapped_face_128[[0, c, y_dst, x_dst]];

                    original_frame[[0, c, y0, x0]] = swap_val * alpha + orig_val * (1.0 - alpha);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LandmarkSmoother {
    alpha: f32,
    prev_kps: Option<[[f32; 2]; 5]>,
}

impl LandmarkSmoother {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            prev_kps: None,
        }
    }

    pub fn update(&mut self, kps: &[[f32; 2]; 5]) -> [[f32; 2]; 5] {
        match self.prev_kps {
            None => {
                self.prev_kps = Some(*kps);
                *kps
            }
            Some(prev) => {
                let mut smoothed = [[0.0f32; 2]; 5];
                for i in 0..5 {
                    smoothed[i][0] = self.alpha * kps[i][0] + (1.0 - self.alpha) * prev[i][0];
                    smoothed[i][1] = self.alpha * kps[i][1] + (1.0 - self.alpha) * prev[i][1];
                }
                self.prev_kps = Some(smoothed);
                smoothed
            }
        }
    }

    pub fn reset(&mut self) {
        self.prev_kps = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feathered_mask_center_and_edges() {
        let mask = generate_feathered_mask(128, 0.2);

        assert_eq!(mask[[64, 64]], 1.0);
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[127, 127]], 0.0);
    }

    #[test]
    fn test_landmark_smoother_ema() {
        let mut smoother = LandmarkSmoother::new(0.5);

        let kps1 = [[10.0, 10.0]; 5];
        let kps2 = [[20.0, 20.0]; 5];

        let res1 = smoother.update(&kps1);
        assert_eq!(res1, kps1);

        let res2 = smoother.update(&kps2);
        assert!((res2[0][0] - 15.0).abs() < 1e-4);
    }
}