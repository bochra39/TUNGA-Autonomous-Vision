#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${1:-$PROJECT_ROOT/wechat_qrcode_models}"
BASE_URL="https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode"

mkdir -p "$MODEL_DIR"

for file in detect.prototxt detect.caffemodel sr.prototxt sr.caffemodel; do
    echo "[DOWNLOAD] $file"
    curl -L --fail --retry 3 --connect-timeout 15 \
        "$BASE_URL/$file" -o "$MODEL_DIR/$file"
done

echo "[OK] Models downloaded to: $MODEL_DIR"
echo "[CHECK] OpenCV WeChatQRCode support:"
python3 - <<'PY'
import cv2
print("OpenCV:", cv2.__version__)
ready = hasattr(cv2, "wechat_qrcode_WeChatQRCode") or (
    hasattr(cv2, "wechat_qrcode") and hasattr(cv2.wechat_qrcode, "WeChatQRCode")
)
print("WeChatQRCode available:", ready)
if not ready:
    print("WARNING: OpenCV contrib WeChatQRCode support was not detected.")
    print("On ROS systems, verify cv_bridge/OpenCV compatibility before replacing OpenCV.")
PY
