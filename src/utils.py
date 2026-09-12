from pathlib import Path
import cv2
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_text(frame, text, x, y, color=(0, 0, 255), scale=0.7, thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_center_marker(frame, frame_w, frame_h):
    center = (frame_w // 2, frame_h // 2)

    cv2.drawMarker(
        frame,
        center,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=25,
        thickness=2,
    )

    return center
