use crate::inference::types::BoundingBox;

pub fn calculate_iou(box_a: &BoundingBox, box_b: &BoundingBox) -> f32 {
    let x_a = box_a.x1.max(box_b.x1);
    let y_a = box_a.y1.max(box_b.y1);
    let x_b = box_a.x2.min(box_b.x2);
    let y_b = box_a.y2.min(box_b.y2);

    let inter_area = (x_b - x_a).max(0.0) * (y_b - y_a).max(0.0);
    if inter_area == 0.0 {
        return 0.0;
    }

    let area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1);
    let area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1);

    inter_area / (area_a + area_b - inter_area)
}

pub fn non_max_suppression(mut boxes: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();

    while !boxes.is_empty() {
        let current = boxes.remove(0);
        boxes.retain(|b| calculate_iou(&current, b) <= iou_threshold);
        keep.push(current);
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::BoundingBox;

    #[test]
    fn test_iou_calculation() {
        let box_a = BoundingBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 0.9 };
        let box_b = BoundingBox { x1: 5.0, y1: 0.0, x2: 15.0, y2: 10.0, score: 0.8 };

        let iou = calculate_iou(&box_a, &box_b);
        assert!((iou - 0.3333333).abs() < 1e-4);
    }

    #[test]
    fn test_nms_suppresses_overlapping_boxes() {
        let box1 = BoundingBox { x1: 10.0, y1: 10.0, x2: 100.0, y2: 100.0, score: 0.95 };
        let box2 = BoundingBox { x1: 12.0, y1: 12.0, x2: 102.0, y2: 102.0, score: 0.80 }; // Дубль
        
        let box3 = BoundingBox { x1: 300.0, y1: 300.0, x2: 400.0, y2: 400.0, score: 0.85 };

        let boxes = vec![box1, box2, box3];
        let filtered = non_max_suppression(boxes, 0.4);

        assert_eq!(filtered.len(), 2);
        assert!((filtered[0].score - 0.95).abs() < f32::EPSILON);
        assert!((filtered[1].score - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nms_empty_input() {
        let boxes: Vec<BoundingBox> = vec![];
        let filtered = non_max_suppression(boxes, 0.4);
        assert!(filtered.is_empty());
    }
}