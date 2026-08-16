use ndarray::Array4;

pub const REFERENCE_LANDMARKS_128: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// [ m[0]  m[1]  m[2] ]
/// [ m[3]  m[4]  m[5] ]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMatrix2x3 {
    pub m: [f32; 6],
}

impl AffineMatrix2x3 {
    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    pub fn transform_point(&self, point: [f32; 2]) -> [f32; 2] {
        let x = point[0];
        let y = point[1];
        let nx = self.m[0] * x + self.m[1] * y + self.m[2];
        let ny = self.m[3] * x + self.m[4] * y + self.m[5];
        [nx, ny]
    }

    pub fn invert(&self) -> Option<Self> {
        let det = self.m[0] * self.m[4] - self.m[1] * self.m[3];
        if det.abs() < 1e-7 {
            return None;
        }

        let inv_det = 1.0 / det;

        let a = self.m[4] * inv_det;
        let b = -self.m[1] * inv_det;
        let c = -self.m[3] * inv_det;
        let d = self.m[0] * inv_det;

        let tx = -(a * self.m[2] + b * self.m[5]);
        let ty = -(c * self.m[2] + d * self.m[5]);

        Some(Self {
            m: [a, b, tx, c, d, ty],
        })
    }
}

pub fn estimate_similarity_transform(src_kps: &[[f32; 2]; 5]) -> Option<AffineMatrix2x3> {
    let dst_kps = &REFERENCE_LANDMARKS_128;

    let mut src_mean = [0.0f32; 2];
    let mut dst_mean = [0.0f32; 2];

    for i in 0..5 {
        src_mean[0] += src_kps[i][0];
        src_mean[1] += src_kps[i][1];
        dst_mean[0] += dst_kps[i][0];
        dst_mean[1] += dst_kps[i][1];
    }

    src_mean[0] /= 5.0;
    src_mean[1] /= 5.0;
    dst_mean[0] /= 5.0;
    dst_mean[1] /= 5.0;

    let mut src_var = 0.0f32;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    let mut syx = 0.0f32;
    let mut syy = 0.0f32;

    for i in 0..5 {
        let sx = src_kps[i][0] - src_mean[0];
        let sy = src_kps[i][1] - src_mean[1];
        let dx = dst_kps[i][0] - dst_mean[0];
        let dy = dst_kps[i][1] - dst_mean[1];

        src_var += sx * sx + sy * sy;

        sxx += dx * sx;
        sxy += dx * sy;
        syx += dy * sx;
        syy += dy * sy;
    }

    if src_var < 1e-6 {
        return None;
    }

    let a_val = sxx + syy;
    let b_val = sxy - syx;

    let norm = (a_val * a_val + b_val * b_val).sqrt();
    if norm < 1e-6 {
        return None;
    }

    let cos_t = a_val / norm;
    let sin_t = b_val / norm;

    let scale = norm / src_var;

    let a = scale * cos_t;
    let b = scale * sin_t;
    let c = -scale * sin_t;
    let d = scale * cos_t;

    let tx = dst_mean[0] - (a * src_mean[0] + b * src_mean[1]);
    let ty = dst_mean[1] - (c * src_mean[0] + d * src_mean[1]);

    Some(AffineMatrix2x3 {
        m: [a, b, tx, c, d, ty],
    })
}

pub fn warp_affine_128(
    input: &Array4<f32>,
    matrix: &AffineMatrix2x3,
) -> Option<Array4<f32>> {
    let inv_mat = matrix.invert()?;
    let h_src = input.shape()[2] as f32;
    let w_src = input.shape()[3] as f32;

    let mut output = Array4::<f32>::zeros((1, 3, 128, 128));

    for y_dst in 0..128 {
        for x_dst in 0..128 {
            let p_dst = [x_dst as f32, y_dst as f32];
            let p_src = inv_mat.transform_point(p_dst);

            let x_src = p_src[0];
            let y_src = p_src[1];

            if x_src >= 0.0 && x_src < w_src - 1.0 && y_src >= 0.0 && y_src < h_src - 1.0 {
                let x0 = x_src.floor() as usize;
                let y0 = y_src.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let dx = x_src - x0 as f32;
                let dy = y_src - y0 as f32;

                let w00 = (1.0 - dx) * (1.0 - dy);
                let w01 = dx * (1.0 - dy);
                let w10 = (1.0 - dx) * dy;
                let w11 = dx * dy;

                for c in 0..3 {
                    let v00 = input[[0, c, y0, x0]];
                    let v01 = input[[0, c, y0, x1]];
                    let v10 = input[[0, c, y1, x0]];
                    let v11 = input[[0, c, y1, x1]];

                    output[[0, c, y_dst, x_dst]] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                }
            }
        }
    }

    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let matrix = AffineMatrix2x3::identity();
        let point = [100.0, 50.0];
        let transformed = matrix.transform_point(point);

        assert_eq!(point, transformed);
    }

    #[test]
    fn test_matrix_inversion() {
        let matrix = AffineMatrix2x3 {
            m: [2.0, 0.5, 10.0, -0.5, 1.5, 5.0],
        };

        let inv = matrix.invert().expect("Matrix should be invertible");
        let point = [30.0, 40.0];
        let transformed = matrix.transform_point(point);
        let restored = inv.transform_point(transformed);

        assert!((point[0] - restored[0]).abs() < 1e-4);
        assert!((point[1] - restored[1]).abs() < 1e-4);
    }

    #[test]
    fn test_similarity_transform_exact_reference() {
        let transform = estimate_similarity_transform(&REFERENCE_LANDMARKS_128)
            .expect("Transform estimation should succeed");

        for i in 0..5 {
            let transformed = transform.transform_point(REFERENCE_LANDMARKS_128[i]);
            assert!((transformed[0] - REFERENCE_LANDMARKS_128[i][0]).abs() < 1e-3);
            assert!((transformed[1] - REFERENCE_LANDMARKS_128[i][1]).abs() < 1e-3);
        }
    }
}