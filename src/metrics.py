def calculate_tracking_outputs(target, frame_w, frame_h, prev_error_x=None, prev_error_y=None, dt=0.0):
    x1, y1, x2, y2 = target["xyxy"]

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    normalized_error_x = (center_x - frame_w / 2.0) / (frame_w / 2.0)
    normalized_error_y = (center_y - frame_h / 2.0) / (frame_h / 2.0)

    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    bbox_area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0

    if prev_error_x is not None and prev_error_y is not None and dt > 0:
        target_velocity_x = (normalized_error_x - prev_error_x) / dt
        target_velocity_y = (normalized_error_y - prev_error_y) / dt
    else:
        target_velocity_x = 0.0
        target_velocity_y = 0.0

    return {
        "center_x": center_x,
        "center_y": center_y,
        "normalized_error_x": normalized_error_x,
        "normalized_error_y": normalized_error_y,
        "target_velocity_x": target_velocity_x,
        "target_velocity_y": target_velocity_y,
        "bbox_area_ratio": bbox_area_ratio,
    }
