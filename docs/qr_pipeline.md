# QR Detection Pipeline

The QR mode was optimized for reading a QR target during a real UAV dive while keeping the video backbone responsive.

## Readers

The final integrated version primarily uses:

1. **WeChatQRCode** for robust decoding of small / perspective-distorted QR targets,
2. **pyzbar** as a secondary reader,
3. OpenCV QR/geometric helper logic retained in the integrated pipeline.

The repository includes a setup script for the four WeChatQRCode model files:

- `detect.prototxt`
- `detect.caffemodel`
- `sr.prototxt`
- `sr.caffemodel`

## Target region

The vision frame contains a mission target/hit area. QR search can operate on that area plus a configurable pixel margin, while final acceptance logic can still check the real target-area geometry.

The final combined mode uses the processed 1280 x 720 frame and supports an initial top crop before all vision operations.

## Candidate validation

The integrated code contains multiple filters intended to reduce false QR candidates:

- quadrilateral area ratio,
- convexity,
- aspect ratio,
- side-length ratio,
- angle checks,
- frame-boundary checks,
- internal black/white ratio,
- edge-density score,
- temporal proximity/lock checks,
- multi-frame verification for unreadable candidates.

## Threading strategy

QR processing is separated from frame acquisition so a slower decoder does not own the camera path. In the final combined configuration:

- WeChatQRCode is the main QR worker,
- pyzbar runs as a secondary periodic worker,
- old OpenCV/enhanced worker code is retained but disabled in the final competition configuration,
- the persistent RTSP backbone remains independent from QR processing.

## ROS publish

After a valid QR text is decoded, the system creates the kamikaze mission message using the shared `/server_time` provider and publishes the decoded text with mission timing data.

The final combined competition configuration keeps the older QR-to-MPC target-error publisher disabled; the QR mode is used for QR mission reading/publishing rather than that legacy MPC output path.
