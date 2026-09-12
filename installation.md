# Installation and Runtime Requirements

The final combined application is more than a normal Python script. `pip install -r requirements.txt` installs only the direct Python packages; a full run also depends on system software, ROS, model files and live runtime inputs.

## 1. Python packages

```bash
python3 -m pip install -r requirements.txt
```

Direct Python dependencies:

- `numpy`
- `PyYAML`
- `ultralytics` — provides YOLO inference and ByteTrack integration
- `pyzbar` — Python wrapper used by the QR fallback reader

`torch` and the normal Ultralytics dependency tree are installed by `ultralytics`; no separate ByteTrack package is required.

## 2. OpenCV / WeChatQRCode

The QR pipeline imports `cv2` and its primary reader uses **WeChatQRCode**, which belongs to OpenCV contrib.

Check your current build:

```bash
python3 - <<'PY'
import cv2
print(cv2.__version__)
print(
    hasattr(cv2, "wechat_qrcode_WeChatQRCode") or
    (hasattr(cv2, "wechat_qrcode") and hasattr(cv2.wechat_qrcode, "WeChatQRCode"))
)
PY
```

For a clean non-ROS environment you can install:

```bash
python3 -m pip install -r requirements-opencv.txt
```

For a ROS Noetic machine, do **not** blindly replace an existing OpenCV installation. Keep an OpenCV build that is compatible with the rest of the ROS environment and exposes WeChatQRCode.

## 3. System packages

The program launches FFmpeg processes directly and `pyzbar` needs the native ZBar runtime.

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg libzbar0
```

Verify:

```bash
ffmpeg -version
python3 -c 'from pyzbar import pyzbar; print("pyzbar OK")'
```

## 4. ROS Noetic

The final competition program requires ROS at runtime. It imports `rospy` and `std_msgs.msg.String`, publishes/subscribes to competition topics, and **will not start the combined competition loop without a valid `/server_time` source**.

The shell running the program must have the ROS environment sourced, for example:

```bash
source /opt/ros/noetic/setup.bash
```

Required ROS Python interfaces include:

- `rospy`
- `std_msgs`

A running ROS master and the relevant competition/server nodes are also required for the full flight configuration.

## 5. Runtime files

Before launching `main.py`, provide:

### `config.yaml`

Create it from the example:

```bash
cp config.example.yaml config.yaml
```

### YOLO weights

Place the trained model at the path configured by `weights`, for example:

```text
weights/your_model.pt
```

The competition-trained weights are intentionally not included in the public repository.

### WeChat QR model files

Download them with:

```bash
chmod +x scripts/setup_wechat_qrcode_models.sh
./scripts/setup_wechat_qrcode_models.sh
```

Expected files:

```text
wechat_qrcode_models/
├── detect.prototxt
├── detect.caffemodel
├── sr.prototxt
└── sr.caffemodel
```

## 6. Live runtime inputs

For the final competition configuration, the system also expects:

- a reachable RTSP/H.264 video source configured by `rtsp_url`
- a ROS master
- a valid `/server_time` publisher
- configured ROS consumers/producers if you want the complete competition data flow
- a CUDA-capable environment if `device: 0` is used; otherwise set `device: cpu`

## 7. Environment check

Run:

```bash
python3 scripts/check_environment.py
```

The checker verifies Python imports, FFmpeg, ROS imports, OpenCV/WeChatQRCode support, configuration presence, YOLO weights and WeChat model files.

## What `requirements.txt` does *not* install

`requirements.txt` intentionally does not install:

- ROS Noetic
- FFmpeg
- native ZBar (`libzbar0`)
- your trained YOLO `.pt` file
- the four WeChat QR model files
- the Herelink/RTSP stream
- the `/server_time` ROS publisher

Those are runtime/system requirements rather than ordinary pip dependencies.
