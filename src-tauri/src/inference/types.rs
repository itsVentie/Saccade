#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    pub points: [(f32, f32); 5],
}

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: BoundingBox,
    pub landmarks: Option<FaceLandmarks>,
}