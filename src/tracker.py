import time
from ultralytics import YOLO


class YoloByteTracker:
    """
    YOLO + ByteTrack wrapper.

    Eski API korunur:
        track(frame) -> (track_boxes, yolo_track_ms)

    Combined LOCK kodu icin yeni API:
        track_with_detections(frame) -> (raw_detections, track_boxes, yolo_track_ms)

    raw_detections, Ultralytics ByteTrack callback'i sonucu filtrelemeden ONCE
    YOLO postprocess sonucundan kopyalanir. Boylece ayni frame icin YOLO sadece
    bir kez calisir; LOCK/detection ham YOLO sonucunu, MPC ise ByteTrack ID'lerini
    kullanabilir.
    """

    def __init__(self, weights, conf_threshold=0.50, iou_threshold=0.45, imgsz=640, device="auto"):
        self.model = YOLO(weights)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = None if device == "auto" else device

        self._raw_detections = []

        # Bu callback model.track() ilk kez cagrildiginda eklenen ByteTrack
        # callback'inden ONCE kaydedilir. Predictor.results burada sadece YOLO
        # postprocess/NMS sonucudur; tracker henuz bu listeyi filtrelememistir.
        self.model.add_callback(
            "on_predict_postprocess_end",
            self._capture_raw_detections_before_tracker,
        )

    def _capture_raw_detections_before_tracker(self, predictor):
        self._raw_detections = []

        try:
            results = getattr(predictor, "results", None)
            if not results:
                return

            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return

            xyxy_list = boxes.xyxy.detach().cpu().numpy()
            conf_list = boxes.conf.detach().cpu().numpy()

            if boxes.cls is not None:
                cls_list = boxes.cls.detach().cpu().numpy()
            else:
                cls_list = [0] * len(xyxy_list)

            raw = []
            for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
                x1, y1, x2, y2 = xyxy.astype(float)
                raw.append({
                    "xyxy": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                })

            self._raw_detections = raw
        except Exception:
            # Tracking calismaya devam etsin. Ana kod bu durumda detection listesini
            # bos gorur; ikinci bir YOLO inference baslatilmaz.
            self._raw_detections = []

    def _run_track_once(self, frame):
        self._raw_detections = []
        start_time = time.perf_counter()

        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        end_time = time.perf_counter()
        yolo_track_ms = (end_time - start_time) * 1000.0

        track_boxes = []

        if not results:
            return list(self._raw_detections), track_boxes, yolo_track_ms

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return list(self._raw_detections), track_boxes, yolo_track_ms

        xyxy_list = result.boxes.xyxy.detach().cpu().numpy()
        conf_list = result.boxes.conf.detach().cpu().numpy()
        id_list = result.boxes.id.detach().cpu().numpy().astype(int)

        if result.boxes.cls is not None:
            cls_list = result.boxes.cls.detach().cpu().numpy()
        else:
            cls_list = [0] * len(xyxy_list)

        for xyxy, conf, cls_id, track_id in zip(xyxy_list, conf_list, cls_list, id_list):
            x1, y1, x2, y2 = xyxy.astype(float)

            track_boxes.append({
                "xyxy": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "track_id": int(track_id),
            })

        return list(self._raw_detections), track_boxes, yolo_track_ms

    def track(self, frame):
        """Eski kullanimlari bozmamak icin eski return yapisi aynen korunur."""
        _raw_detections, track_boxes, yolo_track_ms = self._run_track_once(frame)
        return track_boxes, yolo_track_ms

    def track_with_detections(self, frame):
        """Tek YOLO inference'tan ham detection + ByteTrack ID sonucunu birlikte dondurur."""
        return self._run_track_once(frame)
