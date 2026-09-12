#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#kasmiyor thread yontemi eklendi  - MPC yayin eklendi 

"""
QR Tespit ve Okuma Pipeline - RTSP UDP Sürümü
    tek kareket ayari silmek istesen: if len(qr_text) == 1:
Bu kod RTSP/UDP kamera akışından görüntü alır ve raporda anlatılan QR pipeline mantığını uygular:
1) WeChatQRCode modeli ile küçük/eğik QR okuma denemesi
2) OpenCV QRCodeDetector ile hızlı okuma denemesi
3) Başarısızsa pyzbar ile ek okuma denemesi
3) QR adayı bulunursa ROI çıkarımı
4) ROI üzerinde CLAHE ile kontrast iyileştirme
5) ROI üzerinde adaptive threshold ile ikili görüntü iyileştirme
6) Kontur ve köşe analizi ile QR benzeri dörtgen yapı arama
7) Perspektif düzeltme ile QR bölgesini düzleştirme
8) QR içeriği, okuma zamanı ve görev durumunu ROS'a JSON String olarak yayınlama

WeChatQRCode detector ve süper çözünürlük modelleri bu sürümde isteğe bağlı olarak kullanılır.
"""

# ============================================================
# 1) Gerekli kütüphaneler
# ------------------------------------------------------------
# OpenCV görüntü işleme, QRCodeDetector ve ekran gösterimi için kullanılır.
# pyzbar ikinci QR okuyucu olarak kullanılır.
# numpy matris işlemleri için kullanılır.
# rospy ve std_msgs ROS node/publisher yapısı için kullanılır.
# json ile QR verisi, zaman ve görev durumu tek mesaj içinde gönderilir.
# datetime/time okuma zamanı ve cooldown kontrolü için kullanılır.
# ============================================================

import cv2
from pyzbar import pyzbar
import numpy as np
import json
import time
import math
import subprocess
import threading
import queue
import csv
import os
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import signal

from ultralytics import YOLO
from src.utils import load_config, ensure_dir
from src.tracker import YoloByteTracker
from src.target_state import TargetStateManager
from src.metrics import calculate_tracking_outputs

try:
    import rospy
    from std_msgs.msg import String, Float32MultiArray
except Exception:
    rospy = None
    String = None
    Float32MultiArray = None


# ============================================================
# 2) Temel ayarlar
# ------------------------------------------------------------
# RTSP_URL kameradan gelen görüntü akışının adresidir.
# W ve H görüntü çözünürlüğüdür.
# WINDOW_TITLE OpenCV penceresinin ismidir.
# ============================================================

RTSP_URL = "rtsp://127.0.0.1:8554/fpv_stream"

# ============================================================
# 2.1) Kamera girişi, ham kayıt ve tekrar oynatma ayarları
# ------------------------------------------------------------
# INPUT_MODE:
#   "rtsp"  = gerçek kameradan görüntü alır.
#   "video" = daha önce kaydedilmiş videoyu kamera gibi tekrar oynatır.
#
# RECORD_RAW_VIDEO=True olduğunda yalnızca RTSP modunda kameranın orijinal,
# sıkıştırılmış video akışı MKV dosyasına doğrudan kopyalanır (stream copy).
# Bu dosyada QR kutusu, hedef alanı, yazı veya yazılımsal zoom bulunmaz.
# Yeniden kodlama yapılmadığı için CPU yükü çok düşüktür.
# ============================================================
INPUT_MODE = "rtsp"
VIDEO_PATH = ""                    # INPUT_MODE="video" iken kullanılacak MKV/MP4 dosyası
VIDEO_REALTIME = True              # True: kayıt gerçek kamera hızında oynatılır (-re)
VIDEO_LOOP = True                  # True: video bitince baştan başlar

# RECORD_RAW_VIDEO = True            # DEVRE DISI: Eski ham RTSP kayit ayari
RECORD_RAW_VIDEO = False           # QR kodunda video/ham kayit kapali
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
RECORD_FILE_PREFIX = "raw_camera"
active_recording_path = None

# Dalış sırasında uzaktaki QR daha fazla piksel kaplasın diye varsayılan çözünürlük
# 640x480 yerine 1280x720 yapıldı. ROS parametreleriyle tekrar değiştirilebilir.
W, H = 1280, 720
WINDOW_TITLE = "RTSP QR Detection Pipeline - WeChat Stage 2"


# ============================================================
# 3) ROS ve pipeline parametreleri
# ------------------------------------------------------------
# Bu parametreler ROS parametre sunucusundan okunabilir.
# ROS yoksa veya roscore çalışmıyorsa kod yine kamera testi için çalışır ve QR verisini terminale yazar.
# mission_state raporda geçen görev durumunu QR mesajına eklemek için kullanılır.
# roi_margin QR aday bölgesi çıkarılırken kenarlardan biraz pay bırakır.
# perspective_size perspektif düzeltmeden sonra oluşacak kare QR görüntüsünün boyutudur.
# publish_cooldown_s aynı anda sürekli publish yapılmasını engeller.
# ============================================================

USE_ROS = True
QR_TOPIC = "/kamikaze_bilgisi"
MISSION_STATE = "KAMIKAZE_DIVE"
ROI_MARGIN = 20
PERSPECTIVE_SIZE = 300
PUBLISH_COOLDOWN_S = 0.0

# ============================================================
# 3.1) MPC / Smart Gate parametreleri
# ------------------------------------------------------------
# Bu bölüm QR okuma yöntemini değiştirmez. Sadece okunan veya görsel olarak
# güvenilir bulunan QR merkezini MPC'ye hedef hata verisi olarak gönderir.
#
# ÖNEMLİ:
# - /qr_data: QR yazısı okunursa JSON olarak yayınlanır.
# - /qr/target_error: Smart Gate kabul ederse MPC için hata yayınlanır.
# - QR yoksa valid=0 paketi yayınlanmaz; hiç publish yapılmaz.
# ============================================================

MPC_ERROR_TOPIC = "/qr/target_error"

# Yarışma/hedef alanı oranları.
# Varsayılan: görüntünün ortadaki geniş bölgesi hedef alan kabul edilir.
TARGET_LEFT_RATIO = 0.25
TARGET_RIGHT_RATIO = 0.75
TARGET_TOP_RATIO = 0.10
TARGET_BOTTOM_RATIO = 0.90

# False olursa QR hedef alan dışında olsa bile MPC erken düzeltme alabilir.
# True olursa MPC'ye sadece hedef alan içindeki QR gönderilir.
PUBLISH_ONLY_INSIDE = False

# True olursa QR yazısı okunmasa bile, köşeler/geometri güvenilirse MPC'ye hedef gönderilebilir.
PUBLISH_WHEN_UNREADABLE = True

# Score eşikleri.
# readable: QR yazısı okunmuşsa daha düşük score yeterlidir.
# locked_unreadable: daha önce QR kabul edildiyse ve yeni aday yakındaysa orta score yeterlidir.
# search_unreadable: lock yokken ve QR yazısı okunmuyorsa yüksek score + birkaç frame doğrulama gerekir.
READABLE_MIN_SCORE = 45.0
LOCKED_UNREADABLE_MIN_SCORE = 60.0
SEARCH_UNREADABLE_MIN_SCORE = 82.0

# SEARCH modunda okunmayan QR adayını hemen kabul etmemek için birkaç frame doğrulama.
VERIFY_FRAMES_REQUIRED = 2
VERIFY_MAX_JUMP_PX = 80.0

# Temporal lock: son kabul edilen QR'a yakın yeni adaylara güven puanı verir.
MAX_JUMP_PX = 200.0
LOCK_TIMEOUT_S = 0.65

# Geometrik filtreler.
MIN_AREA_RATIO = 0.00025
MAX_AREA_RATIO = 0.35
ASPECT_MIN = 0.35
ASPECT_MAX = 3.0
MAX_SIDE_RATIO = 6.0
FRAME_MARGIN_RATIO = 0.04

# QR iç patch kontrolü: QR adayının içindeki siyah/beyaz oranı ve kenar yoğunluğu.
USE_PATCH_SCORE = True

# Thread/performance ayarları
# Amaç: kamera akışı QR işlemeyi beklemesin.
FRAME_BUFFER_SIZE = 30          # Geçmiş kare hafızası; worker artık eski kuyruğu sırayla tüketmez.
DISPLAY_EVERY_N = 2             # 2 => 30 FPS gelirse ekranda yaklaşık 15 FPS göster
PYZBAR_FULL_EVERY_N = 1         # pyzbar tüm frame üzerinde ağırdır; her 6 işlenen frame'de bir denenir
HEAVY_SCAN_RECENT_FRAMES = 8    # QR adayı çıkınca son kaç frame ağır pipeline ile tekrar taransın
QR_DRAW_TTL_S = 1.0             # Ekranda son QR kutusunu kaç saniye tutalım

# ============================================================
# 3.1.P) Paralel QR worker ayarları (i7-13700H için güçlü profil)
# ------------------------------------------------------------
# Her okuyucu kendi bağımsız thread'inde ve kendi detector nesnesiyle çalışır.
# Worker yavaş kalırsa eski kareleri sıraya dizmez; en yeni kareye atlar.
# Böylece pyzbar yavaşlasa bile OpenCV ve WeChat beklemez.
# ============================================================
USE_PARALLEL_QR_WORKERS = True
OPENCV_WORKER_EVERY_N = 1        # DEVRE DISI: worker thread baslatilmiyor
WECHAT_WORKER_EVERY_N = 1        # AKTIF: WeChat her frame
PYZBAR_WORKER_EVERY_N = 5      # AKTIF: pyzbar her 5 frame
ENHANCED_WORKER_EVERY_N = 3      # DEVRE DISI: worker thread baslatilmiyor

RESULT_QUEUE_SIZE = 48           # Nadir tespit sonuçları için yeterli; frame kuyruğu değildir
RESULT_DEDUP_TTL_S = 1.0         # Aynı frame/aynı sonucu worker'lar arasında tekilleştir
RESULT_MAX_MPC_FRAME_LAG = 6     # 30 FPS'te ~0.2 s'den eski sonucu MPC'ye verme
OPENCV_INTERNAL_THREADS = 2      # Her OpenCV worker'ın iç paralelliğini sınırlayıp taşmayı önler
WORKER_IDLE_WAIT_S = 0.002
WORKER_STATS_LOG_S = 5.0

# ============================================================
# 3.1.L) TEK UÇUŞ KLASÖRÜ / BİRLEŞİK performans logları
# ------------------------------------------------------------
# worker_attempts.csv: Her worker'ın gördüğü/işlediği kare, süre ve atlama bilgisi.
# detections.csv: Bulunan sonuçların dispatcher, dedup, gecikme ve publish kararı.
# system_events.csv: Kamera/FFmpeg ve program yaşam döngüsü olayları.
# summary.json/txt: Uçuş sonunda worker bazlı otomatik özet.
# Log yazımı ayrı thread'de yapılır; QR worker'larını disk I/O ile bekletmez.
# ============================================================
ENABLE_FLIGHT_LOGGING = True
FLIGHT_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flight_logs")  # fallback; combined main outputs/flight_x/logs kullanir
FLIGHT_LOG_QUEUE_SIZE = 30000
FLIGHT_LOG_FLUSH_INTERVAL_S = 1.0
LOG_SCHEDULED_SKIPS = True
LOG_QR_TEXT = True

# ============================================================
# 3.1.1) Model kullanmadan dalış QR güçlendirmeleri
# ------------------------------------------------------------
# 1) Worker her zaman kuyruğun EN YENİ karesini alır ve eskileri bırakır.
# 2) QR'ın beklenildiği merkez bölge ayrıca yüksek çözünürlükte taranır.
# 3) OpenCV aday bulmasa bile belirli aralıklarla CLAHE/threshold/sharpen taraması yapılır.
# 4) Geçmiş kareler içinde merkez ROI'si en keskin olan kareler tekrar denenir.
# ============================================================
USE_LATEST_FRAME_ONLY = True

USE_CENTER_SCAN_ROI = True
CENTER_SCAN_LEFT_RATIO = 0.12
CENTER_SCAN_RIGHT_RATIO = 0.88
CENTER_SCAN_TOP_RATIO = 0.05
CENTER_SCAN_BOTTOM_RATIO = 0.95
CENTER_SCAN_UPSCALE = 1.35

# 0 yapılırsa aday yokken periyodik güçlendirilmiş tarama kapanır.
PERIODIC_ENHANCED_SCAN_EVERY_N = 2
PERIODIC_RECENT_SCAN_FRAMES = 4
PERIODIC_RECENT_TOP_K = 1

# ============================================================
# 3.1.2) WeChatQRCode model katmanı
# ------------------------------------------------------------
# OpenCV-contrib içindeki WeChatQRCode; detector + süper çözünürlük
# modelleriyle küçük, eğik ve düşük çözünürlüklü QR kodlarda klasik
# QRCodeDetector'a ek bir güçlü okuma katmanı sağlar.
# Model dosyaları bulunamazsa kod hata vermez, otomatik fallback yapar.
# ============================================================
USE_WECHAT_QR = True
WECHAT_SCAN_FULL_FRAME = False
WECHAT_SCAN_CENTER_ROI = False
WECHAT_FULL_EVERY_N = 1
WECHAT_CENTER_EVERY_N = 1
TARGET_SCAN_MARGIN_PX = 30          # Referans QR: vurus alani + 30 px okuma margin
WECHAT_QUEUE_WARN_SIZE = 60         # FIFO gecikme uyarisi; frame drop YOK
WECHAT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wechat_qrcode_models")
WECHAT_DETECT_PROTOTXT = os.path.join(WECHAT_MODELS_DIR, "detect.prototxt")
WECHAT_DETECT_CAFFEMODEL = os.path.join(WECHAT_MODELS_DIR, "detect.caffemodel")
WECHAT_SR_PROTOTXT = os.path.join(WECHAT_MODELS_DIR, "sr.prototxt")
WECHAT_SR_CAFFEMODEL = os.path.join(WECHAT_MODELS_DIR, "sr.caffemodel")

# ============================================================
# 3.2) Temiz ekran çizim ayarları
# ------------------------------------------------------------
# Amaç: ekranda sadece anlamlı olaylar çizilsin.
# - QR yazısı gerçekten okunduysa yeşil çizilir.
# - QR yazısı okunmadı ama Smart Gate MPC için geçerli aday kabul ettiyse turuncu çizilir.
# - Reddedilen adaylar, duvar/gölge/yüz gibi alakasız şekiller çizilmez.
# - Ortadaki hedef/vuruş alanı her zaman sabit sarı çizilir.
#
# Renk değiştirmek istersen:
#   Turuncu BGR = (0, 165, 255)
#   Mavi    BGR = (255, 0, 0)
# ============================================================
DRAW_ONLY_VALID_EVENTS = True
DRAW_TARGET_AREA_ALWAYS = True
DRAW_TARGET_AREA_ON_EVENT = False
DRAW_OVERLAY_ALWAYS = False
DRAW_REJECTED_CANDIDATES = False
QR_READ_COLOR = (0, 255, 0)          # Yeşil: QR yazısı hedef alan içinde gerçekten okundu
QR_BLOCKED_COLOR = (0, 0, 0)         # Siyah: alan dışında okunan veya tek karakterlik QR
MPC_VALID_COLOR = (0, 165, 255)      # Turuncu: QR yazısı yok ama MPC için valid görsel aday
REJECTED_COLOR = (0, 0, 255)         # Kırmızı: sadece debug için


# ============================================================
# 4) Görsel kontrol ayarları
# ------------------------------------------------------------
# z tuşu ile zoom seviyesi değişir.
# Bu yazılımsal zoomdur; kameranın fiziksel zoomu değildir.
# Ekranda takip ve test için kullanılır.
# ============================================================

zoom_levels = [1, 2, 3]
zoom_index = 0

qr_pub = None
mpc_error_pub = None
qr_detector = cv2.QRCodeDetector()
wechat_qr_detector = None
wechat_qr_ready = False
wechat_qr_init_attempted = False

# ============================================================
# Thread paylaşımlı durumları
# ------------------------------------------------------------
# Kamera thread'i sürekli frame alır.
# QR worker thread'i frame_queue içinden frame işler.
# Display ana thread'de kalır; böylece imshow donmaz.
# ============================================================

frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)
qr_result_lock = threading.Lock()
mpc_state_lock = threading.Lock()
pyzbar_call_lock = threading.Lock()
stop_event = threading.Event()

result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
# Referans QR ile ayni: WeChat icin ayri, sinirsiz FIFO. QR modunda frame drop YOK.
# Kuyrukta tam frame yerine sadece vurus alani + margin ROI tutulur.
wechat_frame_queue = queue.Queue()
worker_stats_lock = threading.Lock()
worker_stats = {}
last_qr_frame_id = -1

# Referans QR canli performans sayaçlari.
live_perf_lock = threading.Lock()
wechat_processed_times = deque(maxlen=240)
wechat_processed_total = 0
wechat_forced_skip_total = 0
wechat_last_processed_frame_id = 0
wechat_last_delay_ms = 0.0
wechat_last_q_warn_ts = 0.0

# Uçuş log sistemi paylaşımlı durumları.
flight_log_queue = queue.Queue(maxsize=FLIGHT_LOG_QUEUE_SIZE)
flight_log_state_lock = threading.Lock()
flight_log_session = {}
flight_log_dropped_events = 0
flight_log_close_event = threading.Event()

latest_frame = None
latest_frame_id = 0
latest_frame_capture_ts = 0.0
frame_capture_times = deque(maxlen=FRAME_BUFFER_SIZE)
frame_queue = deque(maxlen=FRAME_BUFFER_SIZE)
frame_history = deque(maxlen=FRAME_BUFFER_SIZE)

last_qr_result = None
last_qr_ts = 0.0
last_publish_ts = 0.0
qr_package_sent = False
qr_package_sent_lock = threading.Lock()

# ============================================================
# MPC Smart Gate çalışma durumu
# ------------------------------------------------------------
# Bu değişkenler QR hedefinin zamansal tutarlılığını takip eder.
# - locked: yakın zamanda güvenilir QR kabul edildi mi?
# - last_accept_xy: son kabul edilen QR merkezi
# - pending_xy/count: SEARCH modunda okunmayan adayın birkaç frame sabit kalıp kalmadığı
# ============================================================

mpc_locked = False
last_accept_time = None
last_accept_xy = None
last_accept_score = 0.0

pending_xy = None
pending_count = 0
pending_score = 0.0

last_mpc_log_ts = 0.0


# ============================================================
# 5) FFmpeg ile RTSP/UDP görüntü alma fonksiyonu
# ------------------------------------------------------------
# Bu fonksiyon bilgisayar kamerası yerine RTSP akışını okur.
# ffplay izleme içindir; burada aynı RTSP adresi ffmpeg ile raw BGR frame olarak Python'a alınır.
# QR pipeline tarafında hiçbir değişiklik yapılmaz.
# ============================================================

def make_recording_path():
    """Her FFmpeg oturumu için çakışmayacak zaman damgalı MKV yolu üretir."""
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(RECORDINGS_DIR, f"{RECORD_FILE_PREFIX}_{stamp}.mkv")


def stop_ffmpeg(proc):
    """
    FFmpeg'i düzgün kapatır.

    Önce rawvideo stdout pipe'ı kapatılır; böylece FFmpeg dolu pipe üzerinde
    bloklanmaz. Ardından FFmpeg'in q komutu denenir ve çıktı dosyasını finalize
    etmesi beklenir. Gerekirse terminate/kill fallback uygulanır.
    """
    if proc is None:
        return

    # QR tarafı artık kare okumayacağı için pipe'ı önce kapatıp FFmpeg'i unblock et.
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:
        pass

    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write(b"q\n")
            proc.stdin.flush()
    except Exception:
        pass

    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass


def start_ffmpeg_legacy_qr_disabled():
    """
    DEVRE DISI / LEGACY: Birlesik sistemde ASLA cagrilmaz.
    Eski standalone QR FFmpeg kodu referans icin tutulur.

    Tek FFmpeg işlemiyle iki işi paralel yapar:

    1) Python QR pipeline'ına 1280x720 BGR kareler verir.
    2) RTSP modunda ve RECORD_RAW_VIDEO=True ise aynı kamera paketlerini
       yeniden kodlamadan, çizimsiz/orijinal biçimde MKV dosyasına kaydeder.

    INPUT_MODE="video" olduğunda VIDEO_PATH dosyası gerçek kamera gibi okunur.
    """
    global active_recording_path

    input_mode = str(INPUT_MODE).strip().lower()
    cmd = ["ffmpeg", "-loglevel", "warning"]

    if input_mode == "rtsp":
        cmd += [
            "-fflags", "nobuffer+genpts+igndts+discardcorrupt",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-rtsp_transport", "udp",
            "-analyzeduration", "5000000",
            "-probesize", "5000000",
            "-stimeout", "2000000",
            "-i", RTSP_URL,
        ]

        if RECORD_RAW_VIDEO:
            active_recording_path = make_recording_path()

            # Birinci çıktı: Kameranın sıkıştırılmış video paketlerini doğrudan MKV'ye kopyala.
            # Herhangi bir OpenCV çizimi veya yazılımsal zoom bu çıkışa ulaşmaz.
            cmd += [
                "-map", "0:v:0",
                "-c:v", "copy",
                "-an",
                "-avoid_negative_ts", "make_zero",
                "-f", "matroska",
                active_recording_path,
            ]
            print(f"[RECORD] Ham kamera kaydı başladı: {active_recording_path}")
        else:
            active_recording_path = None
            print("[RECORD] Ham kamera kaydı kapalı.")

    elif input_mode == "video":
        active_recording_path = None
        video_path = os.path.abspath(os.path.expanduser(str(VIDEO_PATH)))

        if not VIDEO_PATH or not os.path.isfile(video_path):
            raise FileNotFoundError(
                f'INPUT_MODE="video" ancak video bulunamadı: {video_path}'
            )

        if VIDEO_REALTIME:
            cmd += ["-re"]
        if VIDEO_LOOP:
            cmd += ["-stream_loop", "-1"]

        cmd += ["-i", video_path]
        print(f"[VIDEO INPUT] Kayıt tekrar oynatılıyor: {video_path}")
        print(f"[VIDEO INPUT] realtime={VIDEO_REALTIME}, loop={VIDEO_LOOP}")

    else:
        raise ValueError(
            f'Geçersiz INPUT_MODE={INPUT_MODE!r}. "rtsp" veya "video" olmalı.'
        )

    # Son çıktı her iki giriş modunda da Python'a ham BGR kare gönderir.
    cmd += [
        "-map", "0:v:0",
        "-vf", f"fps=30,scale={W}:{H}",
        "-c:v", "rawvideo",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )


# ============================================================
# 5) ROS başlatma fonksiyonu
# ------------------------------------------------------------
# Kod ROS içinde çalışıyorsa node başlatılır ve /qr_data topic'i için publisher oluşturulur.
# ROS başlatılamazsa kod kapanmaz; sadece terminal çıktısı ile test devam eder.
# ============================================================

def init_ros_if_available():
    global qr_pub, mpc_error_pub, USE_ROS, QR_TOPIC, MISSION_STATE, ROI_MARGIN, PERSPECTIVE_SIZE, PUBLISH_COOLDOWN_S
    global W, H
    global FRAME_BUFFER_SIZE, DISPLAY_EVERY_N, PYZBAR_FULL_EVERY_N, HEAVY_SCAN_RECENT_FRAMES, QR_DRAW_TTL_S
    global USE_LATEST_FRAME_ONLY, USE_CENTER_SCAN_ROI
    global CENTER_SCAN_LEFT_RATIO, CENTER_SCAN_RIGHT_RATIO, CENTER_SCAN_TOP_RATIO, CENTER_SCAN_BOTTOM_RATIO
    global CENTER_SCAN_UPSCALE, PERIODIC_ENHANCED_SCAN_EVERY_N
    global PERIODIC_RECENT_SCAN_FRAMES, PERIODIC_RECENT_TOP_K
    global USE_WECHAT_QR, WECHAT_SCAN_FULL_FRAME, WECHAT_SCAN_CENTER_ROI, TARGET_SCAN_MARGIN_PX, WECHAT_QUEUE_WARN_SIZE
    global WECHAT_FULL_EVERY_N, WECHAT_CENTER_EVERY_N, WECHAT_MODELS_DIR
    global WECHAT_DETECT_PROTOTXT, WECHAT_DETECT_CAFFEMODEL, WECHAT_SR_PROTOTXT, WECHAT_SR_CAFFEMODEL
    global MPC_ERROR_TOPIC, TARGET_LEFT_RATIO, TARGET_RIGHT_RATIO, TARGET_TOP_RATIO, TARGET_BOTTOM_RATIO
    global PUBLISH_ONLY_INSIDE, PUBLISH_WHEN_UNREADABLE
    global READABLE_MIN_SCORE, LOCKED_UNREADABLE_MIN_SCORE, SEARCH_UNREADABLE_MIN_SCORE
    global VERIFY_FRAMES_REQUIRED, VERIFY_MAX_JUMP_PX, MAX_JUMP_PX, LOCK_TIMEOUT_S
    global MIN_AREA_RATIO, MAX_AREA_RATIO, ASPECT_MIN, ASPECT_MAX, MAX_SIDE_RATIO, FRAME_MARGIN_RATIO, USE_PATCH_SCORE
    global DRAW_ONLY_VALID_EVENTS, DRAW_TARGET_AREA_ALWAYS, DRAW_TARGET_AREA_ON_EVENT, DRAW_OVERLAY_ALWAYS, DRAW_REJECTED_CANDIDATES
    global ENABLE_FLIGHT_LOGGING, FLIGHT_LOGS_DIR, FLIGHT_LOG_QUEUE_SIZE, FLIGHT_LOG_FLUSH_INTERVAL_S
    global LOG_SCHEDULED_SKIPS, LOG_QR_TEXT
    global USE_PARALLEL_QR_WORKERS, OPENCV_WORKER_EVERY_N, WECHAT_WORKER_EVERY_N
    global PYZBAR_WORKER_EVERY_N, ENHANCED_WORKER_EVERY_N, RESULT_QUEUE_SIZE
    global RESULT_DEDUP_TTL_S, RESULT_MAX_MPC_FRAME_LAG, OPENCV_INTERNAL_THREADS
    global WORKER_IDLE_WAIT_S, WORKER_STATS_LOG_S
    global INPUT_MODE, VIDEO_PATH, VIDEO_REALTIME, VIDEO_LOOP
    global RECORD_RAW_VIDEO, RECORDINGS_DIR, RECORD_FILE_PREFIX

    if not USE_ROS or rospy is None:
        USE_ROS = False
        print("[INFO] ROS kullanılmıyor. QR sonuçları terminale yazılacak.")
        print("[INFO] MPC Smart Gate terminal modunda sadece log yazacak.")
        return

    try:
        rospy.init_node("tunga_vision_system", anonymous=True)

        QR_TOPIC = rospy.get_param("~qr_topic", QR_TOPIC)
        MISSION_STATE = rospy.get_param("~mission_state", MISSION_STATE)
        ROI_MARGIN = int(rospy.get_param("~roi_margin", ROI_MARGIN))
        PERSPECTIVE_SIZE = int(rospy.get_param("~perspective_size", PERSPECTIVE_SIZE))
        PUBLISH_COOLDOWN_S = float(rospy.get_param("~publish_cooldown_s", PUBLISH_COOLDOWN_S))

        # Kamera çözünürlüğü node başlamadan önce okunur; FFmpeg ve frame byte hesabı buna göre kurulur.
        W = int(rospy.get_param("~capture_width", W))
        H = int(rospy.get_param("~capture_height", H))

        # Giriş, ham kayıt ve tekrar oynatma ayarları.
        INPUT_MODE = str(rospy.get_param("~input_mode", INPUT_MODE)).strip().lower()
        VIDEO_PATH = os.path.expanduser(str(rospy.get_param("~video_path", VIDEO_PATH)))
        VIDEO_REALTIME = parse_bool(rospy.get_param("~video_realtime", VIDEO_REALTIME))
        VIDEO_LOOP = parse_bool(rospy.get_param("~video_loop", VIDEO_LOOP))
        # RECORD_RAW_VIDEO = parse_bool(rospy.get_param("~record_raw_video", RECORD_RAW_VIDEO))  # DEVRE DISI
        RECORD_RAW_VIDEO = False  # QR tarafinda kayit kesin kapali
        RECORDINGS_DIR = os.path.abspath(os.path.expanduser(str(
            rospy.get_param("~recordings_dir", RECORDINGS_DIR)
        )))
        RECORD_FILE_PREFIX = str(rospy.get_param("~record_file_prefix", RECORD_FILE_PREFIX)).strip() or "raw_camera"

        # MPC / Smart Gate parametreleri ROS parametre sunucusundan da değiştirilebilir.
        MPC_ERROR_TOPIC = rospy.get_param("~mpc_error_topic", MPC_ERROR_TOPIC)

        TARGET_LEFT_RATIO = float(rospy.get_param("~target_left_ratio", TARGET_LEFT_RATIO))
        TARGET_RIGHT_RATIO = float(rospy.get_param("~target_right_ratio", TARGET_RIGHT_RATIO))
        TARGET_TOP_RATIO = float(rospy.get_param("~target_top_ratio", TARGET_TOP_RATIO))
        TARGET_BOTTOM_RATIO = float(rospy.get_param("~target_bottom_ratio", TARGET_BOTTOM_RATIO))

        PUBLISH_ONLY_INSIDE = parse_bool(rospy.get_param("~publish_only_inside", PUBLISH_ONLY_INSIDE))
        PUBLISH_WHEN_UNREADABLE = parse_bool(rospy.get_param("~publish_when_unreadable", PUBLISH_WHEN_UNREADABLE))

        READABLE_MIN_SCORE = float(rospy.get_param("~readable_min_score", READABLE_MIN_SCORE))
        LOCKED_UNREADABLE_MIN_SCORE = float(rospy.get_param("~locked_unreadable_min_score", LOCKED_UNREADABLE_MIN_SCORE))
        SEARCH_UNREADABLE_MIN_SCORE = float(rospy.get_param("~search_unreadable_min_score", SEARCH_UNREADABLE_MIN_SCORE))

        VERIFY_FRAMES_REQUIRED = max(1, int(rospy.get_param("~verify_frames_required", VERIFY_FRAMES_REQUIRED)))
        VERIFY_MAX_JUMP_PX = float(rospy.get_param("~verify_max_jump_px", VERIFY_MAX_JUMP_PX))
        MAX_JUMP_PX = float(rospy.get_param("~max_jump_px", MAX_JUMP_PX))
        LOCK_TIMEOUT_S = float(rospy.get_param("~lock_timeout_s", LOCK_TIMEOUT_S))

        MIN_AREA_RATIO = float(rospy.get_param("~min_area_ratio", MIN_AREA_RATIO))
        MAX_AREA_RATIO = float(rospy.get_param("~max_area_ratio", MAX_AREA_RATIO))
        ASPECT_MIN = float(rospy.get_param("~aspect_min", ASPECT_MIN))
        ASPECT_MAX = float(rospy.get_param("~aspect_max", ASPECT_MAX))
        MAX_SIDE_RATIO = float(rospy.get_param("~max_side_ratio", MAX_SIDE_RATIO))
        FRAME_MARGIN_RATIO = float(rospy.get_param("~frame_margin_ratio", FRAME_MARGIN_RATIO))
        USE_PATCH_SCORE = parse_bool(rospy.get_param("~use_patch_score", USE_PATCH_SCORE))

        FRAME_BUFFER_SIZE = int(rospy.get_param("~frame_buffer_size", FRAME_BUFFER_SIZE))
        DISPLAY_EVERY_N = max(1, int(rospy.get_param("~display_every_n", DISPLAY_EVERY_N)))
        PYZBAR_FULL_EVERY_N = max(0, int(rospy.get_param("~pyzbar_full_every_n", PYZBAR_FULL_EVERY_N)))
        HEAVY_SCAN_RECENT_FRAMES = max(0, int(rospy.get_param("~heavy_scan_recent_frames", HEAVY_SCAN_RECENT_FRAMES)))
        QR_DRAW_TTL_S = float(rospy.get_param("~qr_draw_ttl_s", QR_DRAW_TTL_S))

        USE_PARALLEL_QR_WORKERS = parse_bool(rospy.get_param("~use_parallel_qr_workers", USE_PARALLEL_QR_WORKERS))
        OPENCV_WORKER_EVERY_N = max(1, int(rospy.get_param("~opencv_worker_every_n", OPENCV_WORKER_EVERY_N)))
        WECHAT_WORKER_EVERY_N = max(1, int(rospy.get_param("~wechat_worker_every_n", WECHAT_WORKER_EVERY_N)))
        PYZBAR_WORKER_EVERY_N = max(1, int(rospy.get_param("~pyzbar_worker_every_n", PYZBAR_WORKER_EVERY_N)))
        ENHANCED_WORKER_EVERY_N = max(1, int(rospy.get_param("~enhanced_worker_every_n", ENHANCED_WORKER_EVERY_N)))
        RESULT_QUEUE_SIZE = max(8, int(rospy.get_param("~result_queue_size", RESULT_QUEUE_SIZE)))
        RESULT_DEDUP_TTL_S = max(0.05, float(rospy.get_param("~result_dedup_ttl_s", RESULT_DEDUP_TTL_S)))
        RESULT_MAX_MPC_FRAME_LAG = max(0, int(rospy.get_param("~result_max_mpc_frame_lag", RESULT_MAX_MPC_FRAME_LAG)))
        OPENCV_INTERNAL_THREADS = max(1, int(rospy.get_param("~opencv_internal_threads", OPENCV_INTERNAL_THREADS)))
        WORKER_IDLE_WAIT_S = max(0.0005, float(rospy.get_param("~worker_idle_wait_s", WORKER_IDLE_WAIT_S)))
        WORKER_STATS_LOG_S = max(1.0, float(rospy.get_param("~worker_stats_log_s", WORKER_STATS_LOG_S)))

        ENABLE_FLIGHT_LOGGING = parse_bool(rospy.get_param("~enable_flight_logging", ENABLE_FLIGHT_LOGGING))
        # Birlesik ucus yapisinda log klasoru outputs/flight_<stamp>/logs olarak sabittir.
        # Eski ROS override DEVRE DISI; loglar farkli klasorlere dagilmasin.
        # FLIGHT_LOGS_DIR = os.path.abspath(os.path.expanduser(str(
        #     rospy.get_param("~flight_logs_dir", FLIGHT_LOGS_DIR)
        # )))
        FLIGHT_LOG_QUEUE_SIZE = max(1000, int(rospy.get_param("~flight_log_queue_size", FLIGHT_LOG_QUEUE_SIZE)))
        FLIGHT_LOG_FLUSH_INTERVAL_S = max(0.1, float(rospy.get_param("~flight_log_flush_interval_s", FLIGHT_LOG_FLUSH_INTERVAL_S)))
        LOG_SCHEDULED_SKIPS = parse_bool(rospy.get_param("~log_scheduled_skips", LOG_SCHEDULED_SKIPS))
        LOG_QR_TEXT = parse_bool(rospy.get_param("~log_qr_text", LOG_QR_TEXT))

        USE_LATEST_FRAME_ONLY = parse_bool(rospy.get_param("~use_latest_frame_only", USE_LATEST_FRAME_ONLY))
        USE_CENTER_SCAN_ROI = parse_bool(rospy.get_param("~use_center_scan_roi", USE_CENTER_SCAN_ROI))
        CENTER_SCAN_LEFT_RATIO = float(rospy.get_param("~center_scan_left_ratio", CENTER_SCAN_LEFT_RATIO))
        CENTER_SCAN_RIGHT_RATIO = float(rospy.get_param("~center_scan_right_ratio", CENTER_SCAN_RIGHT_RATIO))
        CENTER_SCAN_TOP_RATIO = float(rospy.get_param("~center_scan_top_ratio", CENTER_SCAN_TOP_RATIO))
        CENTER_SCAN_BOTTOM_RATIO = float(rospy.get_param("~center_scan_bottom_ratio", CENTER_SCAN_BOTTOM_RATIO))
        CENTER_SCAN_UPSCALE = max(1.0, float(rospy.get_param("~center_scan_upscale", CENTER_SCAN_UPSCALE)))
        PERIODIC_ENHANCED_SCAN_EVERY_N = max(0, int(rospy.get_param("~periodic_enhanced_scan_every_n", PERIODIC_ENHANCED_SCAN_EVERY_N)))
        PERIODIC_RECENT_SCAN_FRAMES = max(0, int(rospy.get_param("~periodic_recent_scan_frames", PERIODIC_RECENT_SCAN_FRAMES)))
        PERIODIC_RECENT_TOP_K = max(0, int(rospy.get_param("~periodic_recent_top_k", PERIODIC_RECENT_TOP_K)))

        USE_WECHAT_QR = parse_bool(rospy.get_param("~use_wechat_qr", USE_WECHAT_QR))
        WECHAT_SCAN_FULL_FRAME = parse_bool(rospy.get_param("~wechat_scan_full_frame", WECHAT_SCAN_FULL_FRAME))
        WECHAT_SCAN_CENTER_ROI = parse_bool(rospy.get_param("~wechat_scan_center_roi", WECHAT_SCAN_CENTER_ROI))
        WECHAT_FULL_EVERY_N = max(0, int(rospy.get_param("~wechat_full_every_n", WECHAT_FULL_EVERY_N)))
        WECHAT_CENTER_EVERY_N = max(0, int(rospy.get_param("~wechat_center_every_n", WECHAT_CENTER_EVERY_N)))
        TARGET_SCAN_MARGIN_PX = max(0, int(rospy.get_param("~target_scan_margin_px", TARGET_SCAN_MARGIN_PX)))
        WECHAT_QUEUE_WARN_SIZE = max(1, int(rospy.get_param("~wechat_queue_warn_size", WECHAT_QUEUE_WARN_SIZE)))
        WECHAT_MODELS_DIR = os.path.expanduser(str(rospy.get_param("~wechat_models_dir", WECHAT_MODELS_DIR)))
        WECHAT_DETECT_PROTOTXT = os.path.join(WECHAT_MODELS_DIR, "detect.prototxt")
        WECHAT_DETECT_CAFFEMODEL = os.path.join(WECHAT_MODELS_DIR, "detect.caffemodel")
        WECHAT_SR_PROTOTXT = os.path.join(WECHAT_MODELS_DIR, "sr.prototxt")
        WECHAT_SR_CAFFEMODEL = os.path.join(WECHAT_MODELS_DIR, "sr.caffemodel")

        # Temiz ekran çizim parametreleri.
        # Varsayılan: sadece QR okunduğunda veya MPC için valid aday oluştuğunda çizim yapılır.
        DRAW_ONLY_VALID_EVENTS = parse_bool(rospy.get_param("~draw_only_valid_events", DRAW_ONLY_VALID_EVENTS))
        DRAW_TARGET_AREA_ALWAYS = parse_bool(rospy.get_param("~draw_target_area_always", DRAW_TARGET_AREA_ALWAYS))
        DRAW_TARGET_AREA_ON_EVENT = parse_bool(rospy.get_param("~draw_target_area_on_event", DRAW_TARGET_AREA_ON_EVENT))
        DRAW_OVERLAY_ALWAYS = parse_bool(rospy.get_param("~draw_overlay_always", DRAW_OVERLAY_ALWAYS))
        DRAW_REJECTED_CANDIDATES = parse_bool(rospy.get_param("~draw_rejected_candidates", DRAW_REJECTED_CANDIDATES))

        qr_pub = rospy.Publisher(QR_TOPIC, String, queue_size=10)

        # MPC TARGET YAYINI DEVRE DISI - eski kod silinmedi, yorumda tutuluyor.
        # if Float32MultiArray is not None:
        #     mpc_error_pub = rospy.Publisher(MPC_ERROR_TOPIC, Float32MultiArray, queue_size=10)
        # else:
        #     mpc_error_pub = None
        #     print("[WARN] Float32MultiArray bulunamadı. MPC topic publish edilemeyecek.")
        mpc_error_pub = None

        print(f"[INFO] ROS aktif. QR Topic: {QR_TOPIC}")
        # print(f"[INFO] MPC Smart Gate Topic: {MPC_ERROR_TOPIC}")  # DEVRE DISI
        print("[INFO] MPC target yayini: DEVRE DISI")
        print(f"[INFO] mission_state={MISSION_STATE}, roi_margin={ROI_MARGIN}, perspective_size={PERSPECTIVE_SIZE}")
        print(f"[INFO] target_roi ratios: left={TARGET_LEFT_RATIO}, right={TARGET_RIGHT_RATIO}, top={TARGET_TOP_RATIO}, bottom={TARGET_BOTTOM_RATIO}")
        print(f"[INFO] smart gate: publish_only_inside={PUBLISH_ONLY_INSIDE}, publish_when_unreadable={PUBLISH_WHEN_UNREADABLE}")
        print(f"[INFO] scores: readable={READABLE_MIN_SCORE}, locked_unreadable={LOCKED_UNREADABLE_MIN_SCORE}, search_unreadable={SEARCH_UNREADABLE_MIN_SCORE}")
        print(f"[INFO] temporal: lock_timeout_s={LOCK_TIMEOUT_S}, max_jump_px={MAX_JUMP_PX}, verify_frames={VERIFY_FRAMES_REQUIRED}")
        print(f"[INFO] capture: {W}x{H}")
        print(f"[INFO] input: mode={INPUT_MODE}, video_path={VIDEO_PATH or '-'}, realtime={VIDEO_REALTIME}, loop={VIDEO_LOOP}")
        print(f"[INFO] recording: enabled={RECORD_RAW_VIDEO}, dir={RECORDINGS_DIR}, prefix={RECORD_FILE_PREFIX}")
        print(f"[INFO] thread params: frame_buffer_size={FRAME_BUFFER_SIZE}, latest_only={USE_LATEST_FRAME_ONLY}, display_every_n={DISPLAY_EVERY_N}, pyzbar_full_every_n={PYZBAR_FULL_EVERY_N}, heavy_scan_recent_frames={HEAVY_SCAN_RECENT_FRAMES}")
        print(f"[INFO] parallel workers: enabled={USE_PARALLEL_QR_WORKERS}, opencv_n={OPENCV_WORKER_EVERY_N}, wechat_n={WECHAT_WORKER_EVERY_N}, pyzbar_n={PYZBAR_WORKER_EVERY_N}, enhanced_n={ENHANCED_WORKER_EVERY_N}, cv_threads={OPENCV_INTERNAL_THREADS}")
        print(f"[INFO] flight logging: enabled={ENABLE_FLIGHT_LOGGING}, dir={FLIGHT_LOGS_DIR}, queue={FLIGHT_LOG_QUEUE_SIZE}, flush={FLIGHT_LOG_FLUSH_INTERVAL_S}s")
        print(f"[INFO] center scan: enabled={USE_CENTER_SCAN_ROI}, roi=({CENTER_SCAN_LEFT_RATIO},{CENTER_SCAN_TOP_RATIO})-({CENTER_SCAN_RIGHT_RATIO},{CENTER_SCAN_BOTTOM_RATIO}), upscale={CENTER_SCAN_UPSCALE}")
        print(f"[INFO] periodic enhanced scan: every_n={PERIODIC_ENHANCED_SCAN_EVERY_N}, recent_frames={PERIODIC_RECENT_SCAN_FRAMES}, recent_top_k={PERIODIC_RECENT_TOP_K}")
        print(f"[INFO] display: only_valid_events={DRAW_ONLY_VALID_EVENTS}, target_always={DRAW_TARGET_AREA_ALWAYS}, target_on_event={DRAW_TARGET_AREA_ON_EVENT}, overlay_always={DRAW_OVERLAY_ALWAYS}, draw_rejected={DRAW_REJECTED_CANDIDATES}")

    except Exception as e:
        USE_ROS = False
        qr_pub = None
        mpc_error_pub = None
        print(f"[WARN] ROS başlatılamadı. Terminal modu ile devam edilecek. Hata: {e}")

def ros_is_shutdown():
    if USE_ROS and rospy is not None:
        return rospy.is_shutdown()
    return False


def ros_shutdown(reason="Kapatıldı"):
    if USE_ROS and rospy is not None:
        rospy.signal_shutdown(reason)


# ============================================================
# 6) Zaman ve içerik doğrulama yardımcıları
# ------------------------------------------------------------
# Raporda okuma zamanı ROS Node'una aktarılır denildiği için her geçerli QR mesajına UTC timestamp eklenir.
# ============================================================

def parse_bool(value):
    """
    ROS parametrelerinden gelen true/false değerlerini güvenli şekilde bool'a çevirir.
    Webcam sürümündeki mantıkla aynıdır.
    Örnek kabul edilen değerler: true, false, 1, 0, yes, no, on, off.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("1", "true", "yes", "on", "evet")


def now_utc_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


# ============================================================
# 7) ROS/terminal yayın fonksiyonu
# ------------------------------------------------------------
# Raporda QR içeriği, okuma zamanı ve görev durumu ROS Node'una aktarılır denildiği için
# mesaj JSON String formatında hazırlanır.
# Yarışma sunucusuna gönderim bu node'un görevi değildir; haberleşme node'u bu mesajı alıp paketler.
# ============================================================

def publish_qr_result(qr_text, method=None, center=None, points=None):
    global shared_server_time_provider

    qr_text = str(qr_text).strip()

    if not qr_text:
        return False

    if shared_server_time_provider is None:
        print("[QR ERROR] /server_time provider hazir degil. QR paketi yayinlanmadi.")
        return False

    try:
        qr_read_time = shared_server_time_provider.now()
    except Exception as exc:
        print(f"[QR ERROR] Server time alinamadi. Paket yayinlanmadi: {exc}")
        return False

    bitis_time = qr_read_time + timedelta(milliseconds=250)
    baslangic_time = bitis_time - timedelta(milliseconds=2700)

    payload = {
        "kamikazeBaslangicZamani": {
            "saat": baslangic_time.hour,
            "dakika": baslangic_time.minute,
            "saniye": baslangic_time.second,
            "milisaniye": baslangic_time.microsecond // 1000,
        },
        "kamikazeBitisZamani": {
            "saat": bitis_time.hour,
            "dakika": bitis_time.minute,
            "saniye": bitis_time.second,
            "milisaniye": bitis_time.microsecond // 1000,
        },
        "qrMetni": qr_text,
    }

    encoded = json.dumps(payload, ensure_ascii=False)

    if USE_ROS and qr_pub is not None:
        qr_pub.publish(encoded)
        print(f"[QR PUBLISH] {encoded}")
        return True

    print(f"[QR READ] {encoded}")
    return False



# ============================================================
# 7.1) MPC yayın fonksiyonu
# ------------------------------------------------------------
# /qr/target_error topic'i MPC tarafının kullanacağı sayısal hata mesajıdır.
#
# Mesaj sırası:
# [valid_for_mpc, qr_x, qr_y, center_x, center_y,
#  error_x_px, error_y_px, error_x_norm, error_y_norm, distance_px]
#
# Zero packet göndermeme kuralı:
# Bu fonksiyon sadece Smart Gate kabul ederse çağrılır. QR yoksa veya aday
# reddedilirse valid=0 mesajı yayınlanmaz; hiç publish yapılmaz.
# ============================================================

def publish_mpc_error(valid_for_mpc, qr_x, qr_y, center_x, center_y,
                      err_x_px, err_y_px, err_x_norm, err_y_norm, dist_px):
    data = [
        float(valid_for_mpc),
        float(qr_x),
        float(qr_y),
        float(center_x),
        float(center_y),
        float(err_x_px),
        float(err_y_px),
        float(err_x_norm),
        float(err_y_norm),
        float(dist_px),
    ]

    if USE_ROS and mpc_error_pub is not None and Float32MultiArray is not None:
        msg = Float32MultiArray()
        msg.data = data
        mpc_error_pub.publish(msg)
    else:
        print(f"[MPC TARGET ERROR] {data}")


# ============================================================
# 7.2) Hedef alan / yarışma ROI fonksiyonları
# ------------------------------------------------------------
# Bu ROI, QR okuma için kullanılan küçük ROI değildir.
# Bu ROI, görüntü içinde MPC'nin hedef merkezi olarak kullanacağı yarışma alanıdır.
# Varsayılan olarak görüntünün ortadaki bölgesi hedef kabul edilir.
# ============================================================

def get_target_area(frame_w, frame_h):
    x1 = int(frame_w * TARGET_LEFT_RATIO)
    x2 = int(frame_w * TARGET_RIGHT_RATIO)
    y1 = int(frame_h * TARGET_TOP_RATIO)
    y2 = int(frame_h * TARGET_BOTTOM_RATIO)

    # Yanlış parametre verilirse güvenli hale getir.
    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(0, min(frame_w - 1, x2))
    y1 = max(0, min(frame_h - 1, y1))
    y2 = max(0, min(frame_h - 1, y2))

    if x2 <= x1:
        x1, x2 = int(frame_w * 0.25), int(frame_w * 0.75)
    if y2 <= y1:
        y1, y2 = int(frame_h * 0.10), int(frame_h * 0.90)

    return x1, y1, x2, y2




def get_target_scan_roi(frame, margin_px=None):
    """WeChat/pyzbar taramasi icin vurus alani + margin ROI dondurur.

    Bu sadece OKUMA alanidir. /qr_data kabul sarti degismez:
    QR'in butun noktalarinin gercek get_target_area() icinde olmasi gerekir.
    """
    if frame is None:
        return None, (0, 0)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_target_area(w, h)
    margin = TARGET_SCAN_MARGIN_PX if margin_px is None else int(margin_px)
    margin = max(0, margin)

    sx1 = max(0, x1 - margin)
    sy1 = max(0, y1 - margin)
    sx2 = min(w, x2 + margin)
    sy2 = min(h, y2 + margin)

    if sx2 <= sx1 or sy2 <= sy1:
        return None, (0, 0)

    return frame[sy1:sy2, sx1:sx2].copy(), (sx1, sy1)

def target_area_center(target_roi):
    x1, y1, x2, y2 = target_roi
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def all_points_inside_target_roi(points, target_roi):
    if points is None:
        return False

    x1, y1, x2, y2 = target_roi
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    for p in pts:
        x = int(p[0])
        y = int(p[1])
        if x < x1 or x > x2 or y < y1 or y > y2:
            return False

    return True


def draw_target_area(frame, color=(0, 255, 255)):
    """
    Ekranda hedef/vuruş alanını ve merkez noktasını gösterir.

    Temiz ekran mantığı:
    - Normalde sürekli çizilmez.
    - QR okunduğunda veya MPC için geçerli aday oluştuğunda kısa süre çizilir.
    - color parametresi ile olay türüne göre sarı/turuncu/yeşil gösterilebilir.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_target_area(w, h)
    cx, cy = target_area_center((x1, y1, x2, y2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 30, 2)
    cv2.putText(frame, "MPC TARGET ROI", (x1 + 8, max(y1 - 8, 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ============================================================
# 7.3) Patch score
# ------------------------------------------------------------
# Amaç: QR'a benzeyen ama aslında zemin/gölge/çatı gibi yanlış dörtgenleri elemek.
#
# Yöntem:
# 1) QR adayı perspektif düzeltme ile kare patch'e çevrilir.
# 2) Adaptive threshold ile siyah/beyaz yapı çıkarılır.
# 3) black_ratio: patch içindeki siyah piksel oranı
# 4) edge_ratio: patch içindeki kenar yoğunluğu
#
# Gerçek QR genelde tamamen beyaz/siyah değildir ve çok sayıda keskin kenar taşır.
# ============================================================

def patch_score(frame, points):
    try:
        rect = order_points(points)
        if rect is None:
            return 0.0, 0.0, 0.0

        size = 120
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (size, size))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )

        black_ratio = float(np.mean(th < 128))

        edges = cv2.Canny(gray, 60, 160)
        edge_ratio = float(np.mean(edges > 0))

        score = 0.0

        # QR ne tamamen beyaz ne tamamen siyah olur.
        if 0.15 <= black_ratio <= 0.75:
            score += 10.0

        # QR içinde çok sayıda keskin geçiş/kenar bulunur.
        if edge_ratio >= 0.035:
            score += 10.0

        return score, black_ratio, edge_ratio

    except Exception:
        return 0.0, 0.0, 0.0


# ============================================================
# 7.4) Geometrik kalite skoru
# ------------------------------------------------------------
# QR adayı sadece "dört nokta bulundu" diye MPC'ye gönderilmez.
# Önce şekil mantıklı mı diye puanlanır:
# - alan oranı
# - convex dörtgen olması
# - aspect ratio
# - kenar oranı
# - açıların aşırı bozuk olmaması
# - frame sınırları içinde olması
# - patch score
# ============================================================

def angle_deg(a, b, c):
    ab = a - b
    cb = c - b

    nab = np.linalg.norm(ab)
    ncb = np.linalg.norm(cb)

    if nab < 1e-6 or ncb < 1e-6:
        return 0.0

    cosang = np.dot(ab, cb) / (nab * ncb)
    cosang = max(-1.0, min(1.0, float(cosang)))
    return math.degrees(math.acos(cosang))


def quad_quality_score(frame, points):
    h, w = frame.shape[:2]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(pts) < 4:
        return 0.0, {
            "area_ratio": 0.0,
            "aspect": 0.0,
            "convex": False,
            "side_ratio": 999999.0,
            "angle_ok_count": 0,
            "frame_ok": False,
            "patch_score": 0.0,
            "black_ratio": 0.0,
            "edge_ratio": 0.0,
        }

    rect = order_points(pts)
    if rect is None:
        return 0.0, {
            "area_ratio": 0.0,
            "aspect": 0.0,
            "convex": False,
            "side_ratio": 999999.0,
            "angle_ok_count": 0,
            "frame_ok": False,
            "patch_score": 0.0,
            "black_ratio": 0.0,
            "edge_ratio": 0.0,
        }

    frame_area = float(w * h)
    area = abs(float(cv2.contourArea(rect)))
    area_ratio = area / max(frame_area, 1.0)

    x, y, bw, bh = cv2.boundingRect(rect.astype(np.int32))
    aspect = float(bw) / max(float(bh), 1.0)

    convex = cv2.isContourConvex(rect.astype(np.int32))

    margin_x = w * FRAME_MARGIN_RATIO
    margin_y = h * FRAME_MARGIN_RATIO

    frame_ok = True
    for p in rect:
        if p[0] < -margin_x or p[0] > w + margin_x or p[1] < -margin_y or p[1] > h + margin_y:
            frame_ok = False
            break

    sides = []
    for i in range(4):
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        sides.append(float(np.linalg.norm(p2 - p1)))

    min_side = min(sides)
    max_side = max(sides)
    side_ratio = max_side / max(min_side, 1.0)

    angles = []
    for i in range(4):
        prev_p = rect[(i - 1) % 4]
        cur_p = rect[i]
        next_p = rect[(i + 1) % 4]
        angles.append(angle_deg(prev_p, cur_p, next_p))

    angle_ok_count = 0
    for a in angles:
        if 35.0 <= a <= 145.0:
            angle_ok_count += 1

    score = 0.0

    area_ok = MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
    if area_ok:
        score += 15.0

    if convex:
        score += 15.0

    if ASPECT_MIN <= aspect <= ASPECT_MAX:
        score += 12.0

    if min_side >= 8.0 and side_ratio <= MAX_SIDE_RATIO:
        score += 12.0

    if angle_ok_count >= 3:
        score += 12.0

    if frame_ok:
        score += 10.0

    patch_s = 0.0
    black_ratio = 0.0
    edge_ratio = 0.0

    if USE_PATCH_SCORE:
        patch_s, black_ratio, edge_ratio = patch_score(frame, rect)
        score += patch_s

    details = {
        "area_ratio": area_ratio,
        "aspect": aspect,
        "convex": convex,
        "side_ratio": side_ratio,
        "angle_ok_count": angle_ok_count,
        "frame_ok": frame_ok,
        "patch_score": patch_s,
        "black_ratio": black_ratio,
        "edge_ratio": edge_ratio,
    }

    return score, details


# ============================================================
# 7.5) Temporal lock ve SEARCH verify
# ------------------------------------------------------------
# Temporal lock:
# Son kabul edilen QR'a yakın yeni adaylara güven puanı ekler.
#
# SEARCH verify:
# Lock yokken QR yazısı okunmayan bir aday tek frame'de kabul edilmez.
# Aynı aday birkaç frame boyunca yakın konumda kalırsa kabul edilir.
# ============================================================

def reset_verify_state():
    global pending_xy, pending_count, pending_score
    pending_xy = None
    pending_count = 0
    pending_score = 0.0


def temporal_score(qr_x, qr_y):
    global mpc_locked, last_accept_time, last_accept_xy

    if not mpc_locked or last_accept_xy is None or last_accept_time is None:
        return 0.0, False, 999999.0

    now = time.monotonic()
    age = now - last_accept_time

    if age > LOCK_TIMEOUT_S:
        mpc_locked = False
        return 0.0, False, 999999.0

    dx = qr_x - last_accept_xy[0]
    dy = qr_y - last_accept_xy[1]
    d = math.sqrt(dx * dx + dy * dy)

    if d <= MAX_JUMP_PX:
        # Hedef son kabul edilen noktaya ne kadar yakınsa o kadar yüksek ek puan alır.
        closeness = 1.0 - min(d / max(MAX_JUMP_PX, 1.0), 1.0)
        return 20.0 * closeness, True, d

    return 0.0, False, d


def update_verify_state(qr_x, qr_y, score):
    global pending_xy, pending_count, pending_score

    if pending_xy is None:
        pending_xy = (qr_x, qr_y)
        pending_count = 1
        pending_score = score
        return pending_count

    dx = qr_x - pending_xy[0]
    dy = qr_y - pending_xy[1]
    d = math.sqrt(dx * dx + dy * dy)

    if d <= VERIFY_MAX_JUMP_PX:
        pending_count += 1
        pending_xy = (qr_x, qr_y)
        pending_score = 0.7 * pending_score + 0.3 * score
    else:
        pending_xy = (qr_x, qr_y)
        pending_count = 1
        pending_score = score

    return pending_count


# ============================================================
# 7.6) Decide Publish / Smart Gate
# ------------------------------------------------------------
# Bu fonksiyon MPC'ye veri gönderilip gönderilmeyeceğine karar verir.
#
# Karar mantığı:
# 1) publish_only_inside=True ise hedef alan dışındaki QR reddedilir.
# 2) QR yazısı okunmuşsa daha düşük score ile kabul edilir.
# 3) QR yazısı okunmamış ama lock varsa ve hedef zıplamamışsa kabul edilebilir.
# 4) QR yazısı okunmamış ve lock yoksa yüksek score + birkaç frame doğrulama gerekir.
# 5) Reddedilirse valid=0 gönderilmez; hiç publish yapılmaz.
# ============================================================

def decide_publish(readable, inside, qr_x, qr_y, score):
    global mpc_locked

    if PUBLISH_ONLY_INSIDE and not inside:
        return False, "outside_target_roi_blocked", 0.0, False, 999999.0, 0

    t_score, temporal_ok, jump_px = temporal_score(qr_x, qr_y)
    total_score = score + t_score

    if readable:
        if total_score >= READABLE_MIN_SCORE:
            reset_verify_state()
            return True, "readable_visual_accept", total_score, temporal_ok, jump_px, 0

        return False, "readable_low_score", total_score, temporal_ok, jump_px, 0

    if not PUBLISH_WHEN_UNREADABLE:
        return False, "unreadable_disabled", total_score, temporal_ok, jump_px, 0

    if mpc_locked and temporal_ok:
        if total_score >= LOCKED_UNREADABLE_MIN_SCORE:
            reset_verify_state()
            return True, "locked_unreadable_accept", total_score, temporal_ok, jump_px, 0

        return False, "locked_unreadable_low_score", total_score, temporal_ok, jump_px, 0

    # SEARCH modu: QR yazısı okunmuyor ve lock yok.
    # Tek frame'de kabul etmiyoruz; birkaç frame yakın kalmalı.
    if score >= SEARCH_UNREADABLE_MIN_SCORE:
        verify_count = update_verify_state(qr_x, qr_y, score)
        if verify_count >= VERIFY_FRAMES_REQUIRED:
            return True, "search_verified_unreadable_accept", total_score, temporal_ok, jump_px, verify_count

        return False, "search_verifying", total_score, temporal_ok, jump_px, verify_count

    reset_verify_state()
    return False, "search_unreadable_rejected", total_score, temporal_ok, jump_px, 0


# ============================================================
# 7.7) QR sonucu -> MPC Smart Gate işleme
# ------------------------------------------------------------
# qr_pipeline sonucunu alır, QR merkezi ve hedef merkezi arasındaki hatayı hesaplar,
# score üretir ve Smart Gate kabul ederse /qr/target_error publish eder.
# ============================================================

def process_mpc_target_result(frame, result):
    global mpc_locked, last_accept_time, last_accept_xy, last_accept_score, last_mpc_log_ts

    if frame is None or not result.get("success"):
        return False

    points = result.get("points")
    if points is None:
        return False

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return False

    ordered = order_points(pts)
    if ordered is None:
        return False

    h, w = frame.shape[:2]

    target_roi = get_target_area(w, h)
    center_x, center_y = target_area_center(target_roi)

    qr_x = int(np.mean(ordered[:, 0]))
    qr_y = int(np.mean(ordered[:, 1]))

    err_x_px = qr_x - center_x
    err_y_px = qr_y - center_y

    err_x_norm = err_x_px / (w / 2.0)
    err_y_norm = err_y_px / (h / 2.0)

    dist_px = math.sqrt(err_x_px ** 2 + err_y_px ** 2)

    inside = all_points_inside_target_roi(ordered, target_roi)

    geom_score, details = quad_quality_score(frame, ordered)

    qr_text = result.get("text", "").strip()
    readable = bool(result.get("readable", bool(qr_text)))

    # QR yazısı okunmuşsa hedefin QR olma ihtimali daha yüksek olduğu için ek güven puanı.
    readable_bonus = 40.0 if readable else 0.0
    base_score = geom_score + readable_bonus

    should_publish, reason, total_score, temporal_ok, jump_px, verify_count = decide_publish(
        readable=readable,
        inside=inside,
        qr_x=qr_x,
        qr_y=qr_y,
        score=base_score,
    )

    # Okunmuş QR alan dışındaysa veya tek karakterse ROS MPC yayını da yapılmaz.
    if readable and not inside:
        should_publish = False
        reason = "readable_outside_target_blocked"
    elif readable and len(qr_text) == 1:
        should_publish = False
        reason = "single_character_blocked"

    # Sonucu display/log için result içine ekliyoruz.
    result["readable"] = readable
    result["target_roi"] = target_roi
    result["target_center"] = [center_x, center_y]
    result["mpc_error_px"] = [int(err_x_px), int(err_y_px)]
    result["mpc_error_norm"] = [float(err_x_norm), float(err_y_norm)]
    result["mpc_distance_px"] = float(dist_px)
    result["mpc_inside_target_roi"] = bool(inside)
    result["mpc_geom_score"] = float(geom_score)
    result["mpc_base_score"] = float(base_score)
    result["mpc_total_score"] = float(total_score)
    result["mpc_reason"] = reason
    result["mpc_published"] = bool(should_publish)
    result["mpc_temporal_ok"] = bool(temporal_ok)
    result["mpc_jump_px"] = float(jump_px)
    result["mpc_verify_count"] = int(verify_count)
    result["mpc_score_details"] = details

    if should_publish:
        publish_mpc_error(
            valid_for_mpc=1.0,
            qr_x=qr_x,
            qr_y=qr_y,
            center_x=center_x,
            center_y=center_y,
            err_x_px=err_x_px,
            err_y_px=err_y_px,
            err_x_norm=err_x_norm,
            err_y_norm=err_y_norm,
            dist_px=dist_px,
        )

        mpc_locked = True
        last_accept_time = time.monotonic()
        last_accept_xy = (qr_x, qr_y)
        last_accept_score = total_score

    # Terminal logu çok şişmesin diye 0.5 saniyede bir yazıyoruz.
    now = time.monotonic()
    if now - last_mpc_log_ts > 0.5:
        last_mpc_log_ts = now
        state_txt = "LOCKED" if mpc_locked else "SEARCH"
        print(
            "[MPC SMART GATE] "
            f"state={state_txt} publish={int(should_publish)} reason={reason} "
            f"readable={int(readable)} inside={int(inside)} "
            f"qr=({qr_x},{qr_y}) err_norm=({err_x_norm:+.3f},{err_y_norm:+.3f}) "
            f"geom={geom_score:.1f} base={base_score:.1f} total={total_score:.1f} "
            f"temporal={int(temporal_ok)} jump={jump_px:.1f} verify={verify_count} "
            f"patch={details.get('patch_score', 0.0):.1f}"
        )

    return should_publish

# ============================================================
# 8) Yazılımsal zoom fonksiyonu
# ------------------------------------------------------------
# Bu fonksiyon frame'in merkezini kırpar ve tekrar ana çözünürlüğe büyütür.
# QR hedef görüntünün merkezine yakınsa test sırasında daha net görünmesini sağlar.
# ============================================================

def optical_zoom(frame, zoom_factor):
    if zoom_factor == 1:
        return frame.copy()

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    nw, nh = w // zoom_factor, h // zoom_factor

    x1 = max(cx - nw // 2, 0)
    y1 = max(cy - nh // 2, 0)
    x2 = min(cx + nw // 2, w)
    y2 = min(cy + nh // 2, h)

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ============================================================
# 8.1) WeChatQRCode model başlatma ve okuma
# ============================================================
def init_wechat_qr_if_available():
    global wechat_qr_detector, wechat_qr_ready, wechat_qr_init_attempted

    if wechat_qr_init_attempted:
        return wechat_qr_ready

    wechat_qr_init_attempted = True
    wechat_qr_ready = False
    wechat_qr_detector = None

    if not USE_WECHAT_QR:
        print("[INFO] WeChatQRCode ROS/ayar parametresi ile kapalı.")
        return False

    required = [
        WECHAT_DETECT_PROTOTXT,
        WECHAT_DETECT_CAFFEMODEL,
        WECHAT_SR_PROTOTXT,
        WECHAT_SR_CAFFEMODEL,
    ]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        print("[WARN] WeChatQRCode model dosyaları eksik; klasik pipeline ile devam edilecek.")
        print(f"[WARN] Beklenen klasör: {WECHAT_MODELS_DIR}")
        for p in missing:
            print(f"[WARN] Eksik: {p}")
        return False

    try:
        constructor = None
        if hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            constructor = cv2.wechat_qrcode_WeChatQRCode
        elif hasattr(cv2, "wechat_qrcode") and hasattr(cv2.wechat_qrcode, "WeChatQRCode"):
            constructor = cv2.wechat_qrcode.WeChatQRCode

        if constructor is None:
            print("[WARN] Bu OpenCV kurulumu WeChatQRCode içermiyor. opencv-contrib-python gerekir.")
            return False

        wechat_qr_detector = constructor(
            WECHAT_DETECT_PROTOTXT,
            WECHAT_DETECT_CAFFEMODEL,
            WECHAT_SR_PROTOTXT,
            WECHAT_SR_CAFFEMODEL,
        )
        wechat_qr_ready = True
        print(f"[INFO] WeChatQRCode aktif. Model klasörü: {WECHAT_MODELS_DIR}")
        return True
    except Exception as e:
        print(f"[WARN] WeChatQRCode başlatılamadı; fallback aktif. Hata: {e}")
        wechat_qr_detector = None
        wechat_qr_ready = False
        return False


def _normalize_wechat_points(points_item):
    if points_item is None:
        return None
    pts = np.asarray(points_item, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    return order_points(pts)


def read_with_wechat_qr(frame, method="wechat_qrcode"):
    if frame is None or frame.size == 0:
        return {"success": False}

    if not init_wechat_qr_if_available():
        return {"success": False}

    try:
        decoded_info, points = wechat_qr_detector.detectAndDecode(frame)
        decoded_info = list(decoded_info) if decoded_info is not None else []
        points = list(points) if points is not None else []

        for idx, text in enumerate(decoded_info):
            text = str(text).strip()
            if not text:
                continue

            pts = _normalize_wechat_points(points[idx] if idx < len(points) else None)
            if pts is None:
                continue

            return {
                "success": True,
                "readable": True,
                "text": text,
                "points": pts,
                "method": method,
            }
    except Exception as e:
        # Sürekli video akışında tek bozuk kare node'u durdurmamalı.
        if not hasattr(read_with_wechat_qr, "last_error_ts"):
            read_with_wechat_qr.last_error_ts = 0.0
        now = time.monotonic()
        if now - read_with_wechat_qr.last_error_ts > 2.0:
            print(f"[WARN] WeChatQRCode frame hatası: {e}")
            read_with_wechat_qr.last_error_ts = now

    return {"success": False}


# ============================================================
# 9) OpenCV QRCodeDetector ilk okuma aşaması
# ------------------------------------------------------------
# Rapordaki ilk aşamadır.
# QRCodeDetector ile hem QR içeriği hem de köşe noktaları alınmaya çalışılır.
# Başarılı olursa sonraki ağır işlemlere gerek kalmadan sonuç döner.
# ============================================================

def read_with_opencv_qr(frame):
    try:
        data, points, _ = qr_detector.detectAndDecode(frame)

        if data and points is not None:
            pts = points.reshape(-1, 2).astype(np.float32)
            return {
                "success": True,
                "text": data.strip(),
                "points": pts,
                "method": "opencv_qrcode_detector",
            }
    except Exception:
        pass

    return {"success": False}


# ============================================================
# 10) pyzbar ikinci okuyucu aşaması
# ------------------------------------------------------------
# Rapora göre OpenCV başarısız olursa pyzbar ikinci okuyucu olarak devreye girer.
# pyzbar QR verisini ve polygon/köşe bilgisini döndürür.
# ============================================================

def read_with_pyzbar(frame):
    try:
        decoded_objects = pyzbar.decode(frame)

        for obj in decoded_objects:
            text = obj.data.decode("utf-8", errors="ignore").strip()
            if not text:
                continue

            polygon = obj.polygon
            if polygon and len(polygon) >= 4:
                pts = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)

                if len(pts) > 4:
                    hull = cv2.convexHull(pts.astype(np.int32))
                    pts = hull.reshape(-1, 2).astype(np.float32)

                return {
                    "success": True,
                    "text": text,
                    "points": pts,
                    "method": "pyzbar_fallback",
                }

            rect = obj.rect
            x, y, w, h = rect.left, rect.top, rect.width, rect.height
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            return {
                "success": True,
                "text": text,
                "points": pts,
                "method": "pyzbar_fallback_rect",
            }

    except Exception:
        pass

    return {"success": False}


# ============================================================
# 11) QR aday noktalarını düzenleme fonksiyonu
# ------------------------------------------------------------
# Perspektif düzeltme için dört noktanın sıralı olması gerekir.
# Bu fonksiyon noktaları sol-üst, sağ-üst, sağ-alt, sol-alt sırasına getirir.
# ============================================================

def order_points(pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)

    if len(pts) < 4:
        return None

    if len(pts) > 4:
        hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2).astype(np.float32)
        pts = hull

    if len(pts) != 4:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        pts = cv2.boxPoints(rect).astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # sol-üst
    ordered[2] = pts[np.argmax(s)]       # sağ-alt
    ordered[1] = pts[np.argmin(diff)]    # sağ-üst
    ordered[3] = pts[np.argmax(diff)]    # sol-alt

    return ordered


# ============================================================
# 12) ROI çıkarımı
# ------------------------------------------------------------
# Raporda işlemlerin tüm görüntü üzerinde değil, QR adayı belirlendiğinde ilgili ROI üzerinde
# yürütüleceği belirtilmiştir. Bu fonksiyon QR köşelerinden ROI keser.
# ============================================================

def extract_roi(frame, points, margin=20):
    if points is None or len(points) < 4:
        return None, None

    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    h, w = frame.shape[:2]

    x_min = max(int(np.min(pts[:, 0])) - margin, 0)
    y_min = max(int(np.min(pts[:, 1])) - margin, 0)
    x_max = min(int(np.max(pts[:, 0])) + margin, w - 1)
    y_max = min(int(np.max(pts[:, 1])) + margin, h - 1)

    if x_max <= x_min or y_max <= y_min:
        return None, None

    roi = frame[y_min:y_max, x_min:x_max].copy()
    offset = (x_min, y_min)
    return roi, offset


# ============================================================
# 13) CLAHE ile ROI görüntü iyileştirme
# ------------------------------------------------------------
# Raporda CLAHE düşük kontrast, ışık değişimi ve zor görüntü koşullarına karşı
# ROI üzerinde uygulanacak iyileştirme adımı olarak verilmiştir.
# ============================================================

def apply_clahe_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ============================================================
# 14) Adaptive threshold ile ROI iyileştirme
# ------------------------------------------------------------
# Raporda adaptive threshold QR siyah-beyaz yapısını güçlendirmek için kullanılmıştır.
# Bu adım da yalnızca ROI üzerinde uygulanır.
# ============================================================

def apply_adaptive_threshold_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


# ============================================================
# 15) Kontur ve köşe analizi
# ------------------------------------------------------------
# Raporda QR geometrik yapısının kontur ve köşe analiziyle çıkarıldığı belirtilmiştir.
# Bu fonksiyon ROI içinde dört köşeli, yeterli büyüklükte QR adayı arar.
# ============================================================

def find_quad_by_contours(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    roi_area = roi.shape[0] * roi.shape[1]

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.05:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return order_points(pts)

    return None


# ============================================================
# 16) Perspektif düzeltme
# ------------------------------------------------------------
# Raporda cv2.getPerspectiveTransform ve cv2.warpPerspective ile perspektif düzeltme
# uygulanacağı yazılmıştır. Bu fonksiyon eğik görünen QR adayını kare görüntüye dönüştürür.
# ============================================================

def perspective_warp(image, points, size=300):
    ordered = order_points(points)
    if ordered is None:
        return None

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped


# ============================================================
# 17) QR aday pipeline fonksiyonu
# ------------------------------------------------------------
# Rapordaki kademeli yapı burada uygulanır.
# Önce tüm görüntüde hızlı okuyucular denenir.
# Aday varsa ROI çıkarılır.
# Ardından ROI üzerinde CLAHE, adaptive threshold, kontur/köşe ve perspektif düzeltme adımları denenir.
# ============================================================

def qr_pipeline(frame):
    # 0. Aşama: WeChatQRCode detector + süper çözünürlük modeli
    result = read_with_wechat_qr(frame, method="wechat_qrcode_full_pipeline")
    if result.get("success"):
        return result

    # 1. Aşama: OpenCV QRCodeDetector
    result = read_with_opencv_qr(frame)
    if result.get("success"):
        return result

    # 2. Aşama: pyzbar fallback
    result = read_with_pyzbar(frame)
    if result.get("success"):
        return result

    # 3. Aşama: QRCodeDetector ile yalnızca aday/köşe tespiti denemesi
    candidate_points = None
    try:
        detected, points = qr_detector.detect(frame)
        if detected and points is not None:
            candidate_points = points.reshape(-1, 2).astype(np.float32)
    except Exception:
        candidate_points = None

    # Eğer hızlı okuyucular QR adayı bile bulamadıysa bu karede işlem sonlandırılır.
    if candidate_points is None:
        return {"success": False}

    # 4. Aşama: ROI çıkarımı
    roi, offset = extract_roi(frame, candidate_points, margin=ROI_MARGIN)
    if roi is None:
        return {"success": False}

    ox, oy = offset
    local_points = candidate_points.copy()
    local_points[:, 0] -= ox
    local_points[:, 1] -= oy

    # 5. Aşama: ROI üzerinde doğrudan tekrar okuma
    result = read_with_opencv_qr(roi)
    if result.get("success"):
        result["method"] = "roi_opencv_qrcode_detector"
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    result = read_with_pyzbar(roi)
    if result.get("success"):
        result["method"] = "roi_pyzbar_fallback"
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    # 6. Aşama: ROI + CLAHE
    clahe_roi = apply_clahe_roi(roi)
    result = read_with_opencv_qr(clahe_roi)
    if result.get("success"):
        result["method"] = "roi_clahe_opencv"
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    result = read_with_pyzbar(clahe_roi)
    if result.get("success"):
        result["method"] = "roi_clahe_pyzbar"
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    # 7. Aşama: ROI + adaptive threshold
    thresh_roi = apply_adaptive_threshold_roi(roi)
    result = read_with_opencv_qr(thresh_roi)
    if result.get("success"):
        result["method"] = "roi_adaptive_threshold_opencv"
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    result = read_with_pyzbar(thresh_roi)
    if result.get("success"):
        result["method"] = "roi_adaptive_threshold_pyzbar"
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    # 8. Aşama: Kontur/köşe analizi ile QR dörtgen adayı bulma
    quad = find_quad_by_contours(thresh_roi)
    if quad is None:
        quad = find_quad_by_contours(clahe_roi)

    if quad is None:
        quad = order_points(local_points)

    if quad is None:
        return {"success": False}

    # 9. Aşama: Perspektif düzeltme
    warped = perspective_warp(roi, quad, size=PERSPECTIVE_SIZE)
    if warped is None:
        return {"success": False}

    result = read_with_opencv_qr(warped)
    if result.get("success"):
        result["method"] = "perspective_warp_opencv"
        result["points"] = quad.copy()
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        return result

    result = read_with_pyzbar(warped)
    if result.get("success"):
        result["method"] = "perspective_warp_pyzbar"
        result["points"] = quad.copy()
        result["points"][:, 0] += ox
        result["points"][:, 1] += oy
        result["readable"] = True
        return result

    # ========================================================
    # MPC için önemli ek:
    # Buraya geldiysek QR yazısı okunamadı; fakat OpenCV QR adayı/köşeleri
    # buldu ve biz bu adaydan güvenilir bir dörtgen çıkardık.
    #
    # Eski qr2 mantığında bu durumda success=False dönerdi.
    # Yeni Smart Gate mantığında ise bu görsel aday MPC için değerlidir.
    # Bu yüzden success=True, readable=False olarak döndürüyoruz.
    # QR yazısı publish edilmez; sadece Smart Gate kabul ederse /qr/target_error gider.
    # ========================================================
    unreadable_points = quad.copy()
    unreadable_points[:, 0] += ox
    unreadable_points[:, 1] += oy

    return {
        "success": True,
        "readable": False,
        "text": "",
        "points": unreadable_points,
        "method": "unreadable_visual_candidate",
    }


# ============================================================
# 18) Görsel çizim fonksiyonu
# ------------------------------------------------------------
# QR bulunduğunda ekranda QR çevresine çizgi çizilir, merkez noktası gösterilir
# ve kullanılan okuma yöntemi ekrana yazılır.
# ============================================================

def should_draw_result(result):
    """
    Ekrana çizim yapılıp yapılmayacağına karar verir.

    İstenen temiz mantık:
    1) QR yazısı gerçekten okunmuşsa çiz.
    2) QR yazısı okunmamış olsa bile Smart Gate MPC için valid aday yayınladıysa çiz.
    3) Reddedilen adayları çizme.
    """
    if result is None:
        return False

    readable = bool(result.get("readable", bool(str(result.get("text", "")).strip())))
    mpc_published = bool(result.get("mpc_published", False))

    if readable:
        return True
    if mpc_published:
        return True

    # Debug istersek reddedilen adayları da açabiliriz. Varsayılan kapalı.
    if DRAW_REJECTED_CANDIDATES and ("mpc_published" in result):
        return True

    return False


def result_draw_color(result):
    """
    Renk ayrımı:
    - Yeşil: QR yazısı hedef alan içinde okundu ve içerik tek karakter değil.
    - Siyah: QR yazısı alan dışında okundu veya içerik tek karakter.
    - Turuncu: QR yazısı okunmadı ama MPC için valid görsel aday var.
    - Kırmızı: sadece debug modunda reddedilen adaylar.
    """
    qr_text = str(result.get("text", "")).strip()
    readable = bool(result.get("readable", bool(qr_text)))
    mpc_published = bool(result.get("mpc_published", False))
    inside = bool(result.get("qr_inside_target_roi", result.get("mpc_inside_target_roi", False)))
    single_character = bool(qr_text) and len(qr_text) == 1

    if readable:
        if not inside:
            return QR_BLOCKED_COLOR, "QR OUTSIDE TARGET"
        if single_character:
            return QR_BLOCKED_COLOR, "QR SINGLE CHAR BLOCKED"
        return QR_READ_COLOR, "QR READ"
    if mpc_published:
        return MPC_VALID_COLOR, "MPC VALID TARGET"
    return REJECTED_COLOR, "REJECTED"


def draw_result(frame, result):
    """
    Temiz çizim fonksiyonu.

    Reddedilen adayları normalde çizmez.
    QR yazısı hedef alan içinde ve tek karakterden uzunsa yeşil çizilir.
    Alan dışında okunan veya tek karakterlik QR siyah; sadece MPC için valid görsel hedef turuncu çizilir.
    """
    if not should_draw_result(result):
        return None

    points = result.get("points")
    text = result.get("text", "")
    method = result.get("method", "")

    color, status = result_draw_color(result)
    center = None

    if points is not None and len(points) >= 4:
        pts = order_points(points)
        if pts is not None:
            pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_i], True, color, 2)

            for p in pts.astype(np.int32):
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, color, -1)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            center = [cx, cy]
            cv2.circle(frame, (cx, cy), 10, color, 2)

            # Renk result_draw_color() tarafından alan/tek-karakter kuralına göre belirlenir.
            # MPC-only durumda metin farklıdır ki QR okundu sanılmasın.
            if text:
                shown_text = "QR READ: " + text
            else:
                shown_text = "MPC VALID VISUAL TARGET"

            cv2.putText(
                frame,
                shown_text,
                (max(cx - 170, 10), max(cy - 45, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2,
            )

            cv2.putText(
                frame,
                status,
                (max(cx - 170, 10), min(cy + 30, frame.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            # Method bilgisi küçük ve beyaz kalsın; ana renklerle karışmasın.
            cv2.putText(
                frame,
                method,
                (max(cx - 170, 10), min(cy + 55, frame.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    return center


# ============================================================
# 19) Üst bilgi yazısı
# ------------------------------------------------------------
# Ekranda RTSP adresi, zoom, görev durumu ve tuş bilgileri gösterilir.
# ============================================================

def draw_overlay(frame):
    text = f"RTSP: {RTSP_URL} | Zoom: {zoom_levels[zoom_index]}x | Mission: {MISSION_STATE} | z: zoom | q: quit"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# ============================================================
# 19.1) Uçuş performans log sistemi
# ============================================================

WORKER_LOG_FIELDS = [
    "utc_time", "session_elapsed_s", "worker", "event", "frame_id",
    "latest_frame_id_start", "latest_frame_id_finish", "forced_skipped_frames",
    "scheduled_every_n", "capture_age_start_ms", "processing_ms",
    "capture_to_finish_ms", "success", "readable", "result_submitted",
    "attempt_profile", "success_method", "qr_text",
]

DETECTION_LOG_FIELDS = [
    "utc_time", "session_elapsed_s", "worker", "frame_id", "method",
    "readable", "qr_text", "camera_fps_at_read", "result_lag_frames", "submit_to_dispatch_ms",
    "capture_to_dispatch_ms", "action", "mpc_allowed", "mpc_published",
    "mpc_reason", "inside_target_roi", "mpc_total_score", "qr_publish_status",
]

SYSTEM_LOG_FIELDS = [
    "utc_time", "session_elapsed_s", "event", "message", "frame_id", "value",
]

RUNTIME_LOG_FIELDS = [
    "utc_time", "session_elapsed_s", "mode", "mode_generation", "camera_frame_id",
    "camera_fps", "wechat_fps", "wechat_queue", "wechat_skipped_total",
    "wechat_lag_frames", "wechat_delay_ms", "lock_queue",
    "backbone_restart_count", "vision_decoder_restart_count",
    "evaluation_written_frames", "evaluation_duplicate_frames",
    "evaluation_deadline_misses",
]


def _flight_utc_now():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _session_elapsed(now=None):
    now = time.monotonic() if now is None else float(now)
    with flight_log_state_lock:
        start = float(flight_log_session.get("start_monotonic", now))
    return max(0.0, now - start)


def _safe_log_text(value):
    text = str(value or "")
    if not LOG_QR_TEXT and text:
        return "<hidden>"
    return text.replace("\n", " ").replace("\r", " ")



def initialize_flight_log_session(session_dir=None, session_id=None):
    """Tek uçuşun TÜM detay loglarını verilen logs/ klasöründe başlatır."""
    global flight_log_queue, flight_log_session, flight_log_dropped_events

    flight_log_queue = queue.Queue(maxsize=max(1000, int(FLIGHT_LOG_QUEUE_SIZE)))
    flight_log_dropped_events = 0
    flight_log_close_event.clear()

    start_mono = time.monotonic()
    if not ENABLE_FLIGHT_LOGGING:
        with flight_log_state_lock:
            flight_log_session = {
                "enabled": False,
                "start_monotonic": start_mono,
                "session_id": str(session_id or "disabled"),
            }
        return None

    if session_dir is None:
        os.makedirs(FLIGHT_LOGS_DIR, exist_ok=True)
        stamp = str(session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        session_dir = os.path.join(FLIGHT_LOGS_DIR, f"flight_{stamp}")
    else:
        stamp = str(session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    session_dir = os.path.abspath(os.path.expanduser(str(session_dir)))
    os.makedirs(session_dir, exist_ok=True)

    session = {
        "enabled": True,
        "session_id": stamp,
        "session_dir": session_dir,
        "start_utc": _flight_utc_now(),
        "start_monotonic": start_mono,
        "worker_csv": os.path.join(session_dir, "qr_worker_attempts.csv"),
        "detection_csv": os.path.join(session_dir, "qr_detections.csv"),
        "system_csv": os.path.join(session_dir, "system_events.csv"),
        "runtime_csv": os.path.join(session_dir, "runtime_metrics.csv"),
        "summary_json": os.path.join(session_dir, "summary.json"),
        "summary_txt": os.path.join(session_dir, "summary.txt"),
    }
    with flight_log_state_lock:
        flight_log_session = session

    print(f"[LOG] Tek uçuş log klasörü: {session_dir}")
    return session_dir


def enqueue_flight_log(kind, row):
    """Disk yazımını worker'dan ayırmak için log olayını kuyruklar."""
    global flight_log_dropped_events
    if not ENABLE_FLIGHT_LOGGING:
        return

    now = time.monotonic()
    payload = dict(row)
    payload.setdefault("utc_time", _flight_utc_now())
    payload.setdefault("session_elapsed_s", round(_session_elapsed(now), 6))

    try:
        flight_log_queue.put_nowait((kind, payload))
    except queue.Full:
        with flight_log_state_lock:
            flight_log_dropped_events += 1


def log_system_event(event, message="", frame_id="", value=""):
    enqueue_flight_log("system", {
        "event": event,
        "message": str(message),
        "frame_id": frame_id,
        "value": value,
    })



def log_runtime_snapshot(frame_id):
    """Q/K/IDLE fark etmeksizin genel FPS, queue ve gecikmeleri tek CSV'ye yazar."""
    cam_fps, wechat_fps, qr_q, skipped, qr_lag, qr_delay_ms = get_live_performance_snapshot()
    try:
        k_q = int(lock_frame_queue.qsize())
    except Exception:
        k_q = 0

    backbone_restarts = int(shared_backbone.restart_count) if shared_backbone is not None else 0
    eval_written = int(shared_evaluation_recorder.total_written_frames) if shared_evaluation_recorder is not None else 0
    eval_duplicates = int(shared_evaluation_recorder.duplicate_frames) if shared_evaluation_recorder is not None else 0
    eval_deadline_misses = int(shared_evaluation_recorder.deadline_misses) if shared_evaluation_recorder is not None else 0

    enqueue_flight_log("runtime", {
        "mode": get_active_mode(),
        "mode_generation": int(get_mode_generation()),
        "camera_frame_id": int(frame_id),
        "camera_fps": round(float(cam_fps), 3),
        "wechat_fps": round(float(wechat_fps), 3),
        "wechat_queue": int(qr_q),
        "wechat_skipped_total": int(skipped),
        "wechat_lag_frames": int(qr_lag),
        "wechat_delay_ms": round(float(qr_delay_ms), 3),
        "lock_queue": int(k_q),
        "backbone_restart_count": backbone_restarts,
        "vision_decoder_restart_count": int(vision_decoder_restart_count),
        "evaluation_written_frames": eval_written,
        "evaluation_duplicate_frames": eval_duplicates,
        "evaluation_deadline_misses": eval_deadline_misses,
    })


def log_worker_attempt(worker, event, frame_id, latest_start, latest_finish,
                       forced_skipped, every_n, capture_ts, started_ts, finished_ts,
                       result=None, submitted=False, attempt_profile=""):
    result = result or {"success": False}
    success = bool(result.get("success"))
    readable = bool(str(result.get("text", "")).strip())
    capture_age_start_ms = ""
    capture_to_finish_ms = ""
    if capture_ts:
        capture_age_start_ms = round(max(0.0, started_ts - capture_ts) * 1000.0, 3)
        capture_to_finish_ms = round(max(0.0, finished_ts - capture_ts) * 1000.0, 3)

    enqueue_flight_log("worker", {
        "worker": worker,
        "event": event,
        "frame_id": int(frame_id),
        "latest_frame_id_start": int(latest_start),
        "latest_frame_id_finish": int(latest_finish),
        "forced_skipped_frames": int(max(0, forced_skipped)),
        "scheduled_every_n": int(max(1, every_n)),
        "capture_age_start_ms": capture_age_start_ms,
        "processing_ms": round(max(0.0, finished_ts - started_ts) * 1000.0, 3),
        "capture_to_finish_ms": capture_to_finish_ms,
        "success": int(success),
        "readable": int(readable),
        "result_submitted": int(bool(submitted)),
        "attempt_profile": attempt_profile,
        "success_method": result.get("method", "") if success else "",
        "qr_text": _safe_log_text(result.get("text", "")) if success else "",
    })


def log_detection_event(worker, frame_id, result, submitted_ts, action,
                        lag_frames="", mpc_allowed="", processed_result=None):
    now = time.monotonic()
    processed = processed_result or result or {}
    capture_ts = float(result.get("capture_monotonic", 0.0) or 0.0)
    capture_to_dispatch_ms = ""
    if capture_ts:
        capture_to_dispatch_ms = round(max(0.0, now - capture_ts) * 1000.0, 3)

    readable = bool(str(result.get("text", "")).strip())
    camera_fps_at_read = get_camera_fps_at_frame(frame_id) if readable else ""

    enqueue_flight_log("detection", {
        "worker": worker,
        "frame_id": int(frame_id),
        "method": result.get("method", ""),
        "readable": int(readable),
        "qr_text": _safe_log_text(result.get("text", "")),
        "camera_fps_at_read": camera_fps_at_read,
        "result_lag_frames": lag_frames,
        "submit_to_dispatch_ms": round(max(0.0, now - submitted_ts) * 1000.0, 3),
        "capture_to_dispatch_ms": capture_to_dispatch_ms,
        "action": action,
        "mpc_allowed": "" if mpc_allowed == "" else int(bool(mpc_allowed)),
        "mpc_published": int(bool(processed.get("mpc_published", False))),
        "mpc_reason": processed.get("mpc_reason", ""),
        "inside_target_roi": "" if "mpc_inside_target_roi" not in processed else int(bool(processed.get("mpc_inside_target_roi"))),
        "mpc_total_score": processed.get("mpc_total_score", ""),
        "qr_publish_status": processed.get("qr_publish_status", ""),
    })


def note_camera_frame(frame_id, capture_ts):
    enqueue_flight_log("camera_frame", {"frame_id": int(frame_id), "capture_ts": float(capture_ts)})




def get_live_camera_fps():
    with frame_lock:
        samples = list(frame_capture_times)
    if len(samples) < 2:
        return 0.0
    dt = float(samples[-1][1]) - float(samples[0][1])
    if dt <= 0.0:
        return 0.0
    return (len(samples) - 1) / dt


def note_wechat_processed(frame_id, finished_ts, capture_ts):
    global wechat_processed_total, wechat_last_processed_frame_id, wechat_last_delay_ms
    with live_perf_lock:
        wechat_processed_times.append(float(finished_ts))
        wechat_processed_total += 1
        wechat_last_processed_frame_id = int(frame_id)
        wechat_last_delay_ms = max(0.0, (float(finished_ts) - float(capture_ts)) * 1000.0)


def get_live_performance_snapshot():
    cam_fps = get_live_camera_fps()
    now = time.monotonic()
    with live_perf_lock:
        recent = [ts for ts in wechat_processed_times if now - ts <= 2.0]
        if len(recent) >= 2:
            dt = recent[-1] - recent[0]
            wechat_fps = (len(recent) - 1) / dt if dt > 0.0 else 0.0
        else:
            wechat_fps = 0.0
        processed_id = int(wechat_last_processed_frame_id)
        skipped = int(wechat_forced_skip_total)
        delay_ms = float(wechat_last_delay_ms)

    with frame_lock:
        current_id = int(latest_frame_id)

    lag_frames = max(0, current_id - processed_id) if processed_id > 0 else 0
    try:
        q_size = int(wechat_frame_queue.qsize())
    except Exception:
        q_size = 0

    return cam_fps, wechat_fps, q_size, skipped, lag_frames, delay_ms


def draw_performance_overlay(frame):
    cam_fps, wechat_fps, q_size, skipped, lag_frames, delay_ms = get_live_performance_snapshot()
    text = (
        f"CAM {cam_fps:.1f} FPS | WECHAT {wechat_fps:.1f} FPS | "
        f"Q {q_size} | SKIP {skipped} | LAG {lag_frames}f | DELAY {delay_ms:.0f} ms"
    )
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def get_frame_capture_ts(frame_id):
    with frame_lock:
        for fid, ts in reversed(frame_capture_times):
            if fid == frame_id:
                return ts
    return 0.0


def get_camera_fps_at_frame(frame_id):
    """QR'ın yakalandığı kareye kadar olan son kamera karelerinden yerel FPS'i hesaplar."""
    with frame_lock:
        samples = [(fid, ts) for fid, ts in frame_capture_times if fid <= int(frame_id)]

    if len(samples) < 2:
        return ""

    # frame_capture_times en fazla FRAME_BUFFER_SIZE kare tuttuğu için bu değer
    # QR'ın okunduğu ana ait yaklaşık son 1 saniyelik gerçek kamera FPS'idir.
    first_ts = float(samples[0][1])
    last_ts = float(samples[-1][1])
    dt = last_ts - first_ts
    if dt <= 0.0:
        return ""

    return round((len(samples) - 1) / dt, 3)


def _new_worker_summary():
    return {
        "observed_events": 0,
        "processed_frames": 0,
        "scheduled_skips": 0,
        "forced_skipped_frames": 0,
        "max_forced_gap": 0,
        "successful_detections": 0,
        "readable_detections": 0,
        "submitted_results": 0,
        "processing_ms_total": 0.0,
        "processing_ms_max": 0.0,
        "capture_to_finish_ms_total": 0.0,
        "capture_to_finish_samples": 0,
        "capture_to_finish_ms_max": 0.0,
        "methods": {},
    }



def _write_flight_summary(summary, session):
    """JSON + okunabilir TXT summary yazar. Finalde unified writer bunu genişletir."""
    workers = summary.get("qr_workers", summary.get("workers", {}))
    for stat in workers.values():
        processed = max(1, stat.get("processed_frames", 0))
        stat["processing_ms_avg"] = round(stat.get("processing_ms_total", 0.0) / processed, 3)
        samples = max(1, stat.get("capture_to_finish_samples", 0))
        stat["capture_to_finish_ms_avg"] = round(stat.get("capture_to_finish_ms_total", 0.0) / samples, 3)
        stat["processing_ms_total"] = round(stat.get("processing_ms_total", 0.0), 3)
        stat["capture_to_finish_ms_total"] = round(stat.get("capture_to_finish_ms_total", 0.0), 3)

    with open(session["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    lines = [
        "TUNGA UNIFIED FLIGHT SUMMARY - PRELIMINARY",
        "=" * 58,
        f"Session: {summary.get('session_id', '-')}",
        f"Start UTC: {summary.get('start_utc', '-')}",
        f"End UTC: {summary.get('end_utc', '-')}",
        f"Duration: {summary.get('duration_s', 0):.3f} s",
        f"Camera frames: {summary.get('camera_frames', 0)}",
        f"Camera avg FPS: {summary.get('camera_fps_avg', 0):.3f}",
        f"Restart events: {summary.get('restart_events', 0)}",
        f"Dropped log events: {summary.get('log_dropped_events', 0)}",
        "",
        "QR WORKERS",
        "-" * 42,
    ]

    for name, stat in workers.items():
        lines.extend([
            f"[{name}]",
            f"  Processed: {stat.get('processed_frames', 0)}",
            f"  Scheduled skips: {stat.get('scheduled_skips', 0)}",
            f"  Forced skipped: {stat.get('forced_skipped_frames', 0)}",
            f"  Successful detection: {stat.get('successful_detections', 0)}",
            f"  Readable QR: {stat.get('readable_detections', 0)}",
            f"  Processing avg/max: {stat.get('processing_ms_avg', 0):.3f} / {stat.get('processing_ms_max', 0):.3f} ms",
            f"  Capture->finish avg/max: {stat.get('capture_to_finish_ms_avg', 0):.3f} / {stat.get('capture_to_finish_ms_max', 0):.3f} ms",
            f"  Methods: {json.dumps(stat.get('methods', {}), ensure_ascii=False)}",
            "",
        ])

    lines.extend([
        "QR PUBLISH STATUS",
        "-" * 42,
        json.dumps(summary.get("qr_publish_status_counts", {}), ensure_ascii=False, indent=2),
        "",
        "GENERAL RUNTIME METRICS",
        "-" * 42,
        json.dumps(summary.get("runtime_metrics", {}), ensure_ascii=False, indent=2),
    ])

    with open(session["summary_txt"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def flight_logger_thread_func():
    """Tüm QR + sistem + genel runtime loglarını tek logs/ klasörüne yazar."""
    with flight_log_state_lock:
        session = dict(flight_log_session)
    if not session.get("enabled"):
        return

    worker_summary = {}
    dispatcher_actions = {}
    qr_publish_status_counts = {}
    camera_frames = 0
    ffmpeg_restarts = 0
    last_flush = time.monotonic()

    runtime_acc = {}
    runtime_count = 0

    def acc_metric(key, value):
        try:
            val = float(value)
        except Exception:
            return
        stat = runtime_acc.setdefault(key, {"sum": 0.0, "max": 0.0, "count": 0})
        stat["sum"] += val
        stat["max"] = max(stat["max"], val)
        stat["count"] += 1

    with open(session["worker_csv"], "w", newline="", encoding="utf-8") as wf, \
         open(session["detection_csv"], "w", newline="", encoding="utf-8") as df, \
         open(session["system_csv"], "w", newline="", encoding="utf-8") as sf, \
         open(session["runtime_csv"], "w", newline="", encoding="utf-8") as rf:
        worker_writer = csv.DictWriter(wf, fieldnames=WORKER_LOG_FIELDS, extrasaction="ignore")
        detection_writer = csv.DictWriter(df, fieldnames=DETECTION_LOG_FIELDS, extrasaction="ignore")
        system_writer = csv.DictWriter(sf, fieldnames=SYSTEM_LOG_FIELDS, extrasaction="ignore")
        runtime_writer = csv.DictWriter(rf, fieldnames=RUNTIME_LOG_FIELDS, extrasaction="ignore")
        worker_writer.writeheader()
        detection_writer.writeheader()
        system_writer.writeheader()
        runtime_writer.writeheader()

        while not flight_log_close_event.is_set() or not flight_log_queue.empty():
            try:
                kind, row = flight_log_queue.get(timeout=0.1)
            except queue.Empty:
                kind = None
                row = None

            if kind == "worker":
                worker_writer.writerow(row)
                name = str(row.get("worker", "unknown"))
                stat = worker_summary.setdefault(name, _new_worker_summary())
                stat["observed_events"] += 1
                forced = int(row.get("forced_skipped_frames", 0) or 0)
                stat["forced_skipped_frames"] += forced
                stat["max_forced_gap"] = max(stat["max_forced_gap"], forced)
                if row.get("event") == "scheduled_skip":
                    stat["scheduled_skips"] += 1
                else:
                    stat["processed_frames"] += 1
                    processing_ms = float(row.get("processing_ms", 0.0) or 0.0)
                    stat["processing_ms_total"] += processing_ms
                    stat["processing_ms_max"] = max(stat["processing_ms_max"], processing_ms)
                    ctf = row.get("capture_to_finish_ms", "")
                    if ctf != "":
                        ctf = float(ctf)
                        stat["capture_to_finish_ms_total"] += ctf
                        stat["capture_to_finish_samples"] += 1
                        stat["capture_to_finish_ms_max"] = max(stat["capture_to_finish_ms_max"], ctf)
                stat["successful_detections"] += int(row.get("success", 0) or 0)
                stat["readable_detections"] += int(row.get("readable", 0) or 0)
                stat["submitted_results"] += int(row.get("result_submitted", 0) or 0)
                method = str(row.get("success_method", "") or "")
                if method:
                    stat["methods"][method] = stat["methods"].get(method, 0) + 1

            elif kind == "detection":
                detection_writer.writerow(row)
                action = str(row.get("action", "unknown"))
                dispatcher_actions[action] = dispatcher_actions.get(action, 0) + 1
                pub_status = str(row.get("qr_publish_status", "") or "")
                if pub_status:
                    qr_publish_status_counts[pub_status] = qr_publish_status_counts.get(pub_status, 0) + 1

            elif kind == "system":
                system_writer.writerow(row)
                if row.get("event") in ("ffmpeg_restart", "backbone_restart", "vision_decoder_restart"):
                    ffmpeg_restarts += 1

            elif kind == "runtime":
                runtime_writer.writerow(row)
                runtime_count += 1
                for key in (
                    "camera_fps", "wechat_fps", "wechat_queue", "wechat_lag_frames",
                    "wechat_delay_ms", "lock_queue", "backbone_restart_count",
                    "vision_decoder_restart_count", "evaluation_written_frames",
                    "evaluation_duplicate_frames", "evaluation_deadline_misses",
                ):
                    acc_metric(key, row.get(key, 0))

            elif kind == "camera_frame":
                camera_frames += 1

            if kind is not None:
                flight_log_queue.task_done()

            now = time.monotonic()
            if now - last_flush >= FLIGHT_LOG_FLUSH_INTERVAL_S:
                wf.flush(); df.flush(); sf.flush(); rf.flush()
                last_flush = now

        wf.flush(); df.flush(); sf.flush(); rf.flush()

    end_mono = time.monotonic()
    duration_s = max(0.0, end_mono - float(session["start_monotonic"]))
    with flight_log_state_lock:
        dropped = int(flight_log_dropped_events)

    runtime_summary = {}
    for key, stat in runtime_acc.items():
        count = max(1, int(stat["count"]))
        runtime_summary[key] = {
            "avg": round(stat["sum"] / count, 3),
            "max": round(stat["max"], 3),
            "samples": int(stat["count"]),
        }

    summary = {
        "session_id": session["session_id"],
        "start_utc": session["start_utc"],
        "end_utc": _flight_utc_now(),
        "duration_s": round(duration_s, 3),
        "camera_frames": int(camera_frames),
        "camera_fps_avg": round(camera_frames / max(duration_s, 1e-6), 3),
        "restart_events": int(ffmpeg_restarts),
        "log_dropped_events": dropped,
        "input_mode": INPUT_MODE,
        "capture_resolution": [int(W), int(H)],
        "worker_every_n": {
            "opencv": int(OPENCV_WORKER_EVERY_N),
            "wechat": int(WECHAT_WORKER_EVERY_N),
            "pyzbar": int(PYZBAR_WORKER_EVERY_N),
            "enhanced": int(ENHANCED_WORKER_EVERY_N),
        },
        "qr_workers": worker_summary,
        "qr_dispatcher_actions": dispatcher_actions,
        "qr_publish_status_counts": qr_publish_status_counts,
        "runtime_metrics": runtime_summary,
        "files": {
            "qr_worker_attempts": session["worker_csv"],
            "qr_detections": session["detection_csv"],
            "system_events": session["system_csv"],
            "runtime_metrics": session["runtime_csv"],
        },
    }
    # Bu ilk özet daha sonra LOCK + video/backbone final değerleriyle genişletilir.
    _write_flight_summary(summary, session)
    print(f"[LOG] Ön uçuş özeti yazıldı: {session['summary_txt']}")


def reset_runtime_buffers():
    global frame_queue, frame_history, result_queue, frame_capture_times, wechat_frame_queue
    global latest_frame, latest_frame_id, latest_frame_capture_ts
    global last_qr_result, last_qr_ts, last_qr_frame_id
    global wechat_processed_times, wechat_processed_total, wechat_forced_skip_total
    global wechat_last_processed_frame_id, wechat_last_delay_ms, wechat_last_q_warn_ts

    with frame_condition:
        frame_queue = deque(maxlen=max(1, int(FRAME_BUFFER_SIZE)))
        frame_history = deque(maxlen=max(1, int(FRAME_BUFFER_SIZE)))
        latest_frame = None
        latest_frame_id = 0
        latest_frame_capture_ts = 0.0
        frame_capture_times = deque(maxlen=max(1, int(FRAME_BUFFER_SIZE)))
        frame_condition.notify_all()

    result_queue = queue.Queue(maxsize=max(8, int(RESULT_QUEUE_SIZE)))
    wechat_frame_queue = queue.Queue()

    with live_perf_lock:
        wechat_processed_times = deque(maxlen=240)
        wechat_processed_total = 0
        wechat_forced_skip_total = 0
        wechat_last_processed_frame_id = 0
        wechat_last_delay_ms = 0.0
        wechat_last_q_warn_ts = 0.0

    with qr_result_lock:
        last_qr_result = None
        last_qr_ts = 0.0
        last_qr_frame_id = -1

    with worker_stats_lock:
        worker_stats.clear()


def compute_center_and_points(result):
    points = result.get("points")
    center = None
    points_list = None

    if points is not None:
        pts = np.array(points).reshape(-1, 2)
        if len(pts) >= 4:
            center = [int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))]
            points_list = [[int(p[0]), int(p[1])] for p in pts]

    return center, points_list



def save_and_publish_qr_result(result, frame=None, frame_id=None, allow_mpc=True):
    """Referans QR yayın mantigi + combined Q/K gecis korumasi."""
    global last_qr_result, last_qr_ts, last_publish_ts, last_qr_frame_id
    global qr_package_sent

    if get_active_mode() != MODE_QR:
        ignored = result.copy() if isinstance(result, dict) else {}
        ignored["qr_publish_status"] = "mode_not_qr"
        return ignored

    result_generation = int(result.get("mode_generation", -1))
    if result_generation != get_mode_generation():
        ignored = result.copy()
        ignored["qr_publish_status"] = "stale_mode_generation"
        return ignored

    qr_text = str(result.get("text", "")).strip()
    now = time.monotonic()
    fid = int(frame_id if frame_id is not None else result.get("frame_id", -1))

    result = result.copy()
    result["frame_id"] = fid
    result["mpc_published"] = False
    result["mpc_reason"] = "mpc_disabled"

    qr_inside_target = False
    points = result.get("points")
    if points is not None:
        if frame is not None:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_w, frame_h = int(W), int(H)
        target_roi = get_target_area(frame_w, frame_h)
        qr_inside_target = all_points_inside_target_roi(points, target_roi)

    result["qr_inside_target_roi"] = bool(qr_inside_target)
    result["qr_single_character"] = bool(qr_text) and len(qr_text) == 1

    with qr_result_lock:
        if fid >= last_qr_frame_id:
            last_qr_result = result.copy()
            last_qr_ts = now
            last_qr_frame_id = fid

    if not qr_text:
        result["qr_publish_status"] = "no_text"
        return result

    if not qr_inside_target:
        print(f"[QR] Hedef alan disinda okundu, /qr_data publish edilmeyecek: {qr_text}")
        result["qr_publish_status"] = "outside_target_blocked"
        return result

    if len(qr_text) == 1:
        print(f"[QR] Tek karakterlik okuma engellendi, /qr_data publish edilmeyecek: {qr_text}")
        result["qr_publish_status"] = "single_character_blocked"
        return result

    with qr_package_sent_lock:
        if qr_package_sent:
            result["qr_publish_status"] = "already_sent_this_q_session"
            return result

        if now - last_publish_ts < PUBLISH_COOLDOWN_S:
            result["qr_publish_status"] = "cooldown_blocked"
            return result

        center, points_list = compute_center_and_points(result)
        published = publish_qr_result(
            qr_text=qr_text,
            method=result.get("method", "unknown"),
            center=center,
            points=points_list,
        )

        if published:
            qr_package_sent = True
            last_publish_ts = now
            result["qr_publish_status"] = "published"
        else:
            result["qr_publish_status"] = "publish_failed"

    return result


def get_center_scan_roi(frame):
    """
    QR'ın dalışta çoğunlukla bulunacağı geniş merkez bölgeyi döndürür.
    Dönüş: roi, (offset_x, offset_y)
    """
    if frame is None:
        return None, (0, 0)

    h, w = frame.shape[:2]

    left = max(0.0, min(1.0, CENTER_SCAN_LEFT_RATIO))
    right = max(0.0, min(1.0, CENTER_SCAN_RIGHT_RATIO))
    top = max(0.0, min(1.0, CENTER_SCAN_TOP_RATIO))
    bottom = max(0.0, min(1.0, CENTER_SCAN_BOTTOM_RATIO))

    if right <= left or bottom <= top:
        left, right, top, bottom = 0.12, 0.88, 0.05, 0.95

    x1 = int(w * left)
    x2 = int(w * right)
    y1 = int(h * top)
    y2 = int(h * bottom)

    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))

    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def prepare_center_scan_image(frame):
    """
    Merkez ROI'yi çıkarır ve küçük QR'ların daha fazla piksel kaplaması için büyütür.
    Dönüş: taranacak görüntü, offset, kullanılan ölçek
    """
    roi, offset = get_center_scan_roi(frame)
    if roi is None or roi.size == 0:
        return None, (0, 0), 1.0

    scale = max(1.0, float(CENTER_SCAN_UPSCALE))
    if abs(scale - 1.0) > 1e-6:
        roi = cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    return roi, offset, scale


def map_result_to_full_frame(result, offset=(0, 0), scale=1.0, method_prefix=None):
    """ROI/ölçek koordinatlarını tam kamera frame koordinatlarına çevirir."""
    if not result or not result.get("success"):
        return result

    mapped = result.copy()
    points = mapped.get("points")

    if points is not None:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        safe_scale = max(float(scale), 1e-6)
        pts[:, 0] = pts[:, 0] / safe_scale + float(offset[0])
        pts[:, 1] = pts[:, 1] / safe_scale + float(offset[1])
        mapped["points"] = pts

    if method_prefix:
        mapped["method"] = f"{method_prefix}_{mapped.get('method', 'unknown')}"

    return mapped


def detect_candidate_points(image):
    """OpenCV ile sadece QR köşe adayı arar; okuyamasa bile noktaları döndürür."""
    try:
        detected, points = qr_detector.detect(image)
        if detected and points is not None:
            return points.reshape(-1, 2).astype(np.float32)
    except Exception:
        pass
    return None


def probe_single_image(image, use_pyzbar=False, use_wechat=False, method_prefix="fast"):
    """Bir görüntüde WeChat/OpenCV ve isteğe bağlı pyzbar taraması yapar."""
    if use_wechat:
        result = read_with_wechat_qr(image, method=f"{method_prefix}_wechat_qrcode")
        if result.get("success"):
            return result, None

    result = read_with_opencv_qr(image)
    if result.get("success"):
        result["method"] = f"{method_prefix}_opencv_qrcode_detector"
        return result, None

    candidate_points = detect_candidate_points(image)

    if use_pyzbar:
        result = read_with_pyzbar(image)
        if result.get("success"):
            result["method"] = f"{method_prefix}_pyzbar"
            return result, candidate_points

    return {"success": False}, candidate_points


def fast_qr_probe(frame, frame_id):
    """
    Hafif tarama:
    1) Tam çözünürlüklü frame üzerinde OpenCV denenir.
    2) Belirli aralıklarla tam frame pyzbar denenir.
    3) Tam frame başarısızsa merkez ROI ayrıca büyütülerek taranır.

    Dönüş:
    - result: okunmuş QR sonucu veya success=False
    - candidate_info: adayın tam frame mi merkez ROI'de mi bulunduğu ve koordinatları
    """
    use_periodic_pyzbar = (
        PYZBAR_FULL_EVERY_N > 0
        and frame_id % PYZBAR_FULL_EVERY_N == 0
    )

    result, full_points = probe_single_image(
        frame,
        use_pyzbar=use_periodic_pyzbar,
        use_wechat=(WECHAT_SCAN_FULL_FRAME and WECHAT_FULL_EVERY_N > 0 and frame_id % WECHAT_FULL_EVERY_N == 0),
        method_prefix="fast_full",
    )
    if result.get("success"):
        return result, None

    candidate_info = None
    if full_points is not None:
        candidate_info = {
            "source": "full",
            "points": full_points,
            "offset": (0, 0),
            "scale": 1.0,
        }

    if USE_CENTER_SCAN_ROI:
        center_image, offset, scale = prepare_center_scan_image(frame)
        if center_image is not None:
            center_result, center_points = probe_single_image(
                center_image,
                use_pyzbar=use_periodic_pyzbar,
                use_wechat=(WECHAT_SCAN_CENTER_ROI and WECHAT_CENTER_EVERY_N > 0 and frame_id % WECHAT_CENTER_EVERY_N == 0),
                method_prefix="fast_center",
            )

            if center_result.get("success"):
                center_result = map_result_to_full_frame(
                    center_result,
                    offset=offset,
                    scale=scale,
                )
                return center_result, None

            if center_points is not None:
                mapped_points = np.asarray(center_points, dtype=np.float32).copy()
                mapped_points[:, 0] = mapped_points[:, 0] / scale + offset[0]
                mapped_points[:, 1] = mapped_points[:, 1] / scale + offset[1]
                candidate_info = {
                    "source": "center",
                    "points": mapped_points,
                    "offset": offset,
                    "scale": scale,
                }

    return {"success": False}, candidate_info


def run_center_qr_pipeline(frame, method_prefix="center_pipeline"):
    """Mevcut ağır QR pipeline'ını büyütülmüş merkez ROI üzerinde çalıştırır."""
    center_image, offset, scale = prepare_center_scan_image(frame)
    if center_image is None:
        return {"success": False}

    result = qr_pipeline(center_image)
    if not result.get("success"):
        return result

    return map_result_to_full_frame(
        result,
        offset=offset,
        scale=scale,
        method_prefix=method_prefix,
    )


def apply_unsharp_mask(image):
    """Hareket/odak yumuşaklığında QR kenarlarını hafifçe belirginleştirir."""
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.8, blurred, -0.8, 0)


def periodic_enhanced_scan(frame):
    """
    OpenCV aday bulmasa bile merkez ROI üzerinde doğrudan decode dener.
    Raw, CLAHE, adaptive threshold ve hafif sharpen varyantları kullanılır.
    """
    scan_image, offset, scale = prepare_center_scan_image(frame)
    if scan_image is None:
        return {"success": False}

    variants = [
        ("raw", scan_image),
        ("clahe", apply_clahe_roi(scan_image)),
        ("threshold", apply_adaptive_threshold_roi(scan_image)),
        ("sharpen", apply_unsharp_mask(scan_image)),
    ]

    for variant_name, variant in variants:
        result = read_with_opencv_qr(variant)
        if result.get("success"):
            result["method"] = f"periodic_{variant_name}_opencv"
            return map_result_to_full_frame(result, offset=offset, scale=scale)

        result = read_with_pyzbar(variant)
        if result.get("success"):
            result["method"] = f"periodic_{variant_name}_pyzbar"
            return map_result_to_full_frame(result, offset=offset, scale=scale)

    return {"success": False}


def center_roi_sharpness(frame):
    """Merkez QR tarama bölgesinin Laplacian keskinlik skorunu döndürür."""
    roi, _ = get_center_scan_roi(frame)
    if roi is None or roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def get_recent_frames(limit, exclude_frame_id=None):
    if limit <= 0:
        return []

    with frame_lock:
        items = list(frame_history)[-limit:]

    if exclude_frame_id is not None:
        items = [(fid, frm) for fid, frm in items if fid != exclude_frame_id]

    return items



def get_sharpest_recent_frames(limit, top_k, exclude_frame_id=None):
    """Son kareler içinden merkez ROI'si en keskin olanları seçer."""
    if limit <= 0 or top_k <= 0:
        return []

    items = get_recent_frames(limit, exclude_frame_id=exclude_frame_id)
    scored = []

    for fid, frame in items:
        try:
            score = center_roi_sharpness(frame)
        except Exception:
            score = 0.0
        scored.append((score, fid, frame))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(fid, frame) for _, fid, frame in scored[:top_k]]



def camera_thread_func():
    """
    ORTAK local vision decoder.

    Herelink'e BAGLANMAZ; tek Herelink RTSP PersistentVideoBackbone'dadir.
    Q modunda her kamera frame'inin vurus alani + 30px margin ROI'si WeChat FIFO'ya gider.
    K modunda mevcut LOCK FIFO/no-drop davranisi aynen korunur.
    """
    global latest_frame, latest_frame_id, latest_frame_capture_ts
    global vision_decoder_restart_count

    frame_bytes = W * H * 3
    proc = start_local_vision_decoder()
    last_capture_ts = None
    last_runtime_log_ts = 0.0

    print("[INFO] Ortak vision decoder thread basladi (localhost relay).")
    log_system_event("camera_start", "shared_local_vision_decoder")

    try:
        while not stop_event.is_set() and not ros_is_shutdown():
            read_start = time.perf_counter()
            raw = _read_exactly_from_proc(proc, frame_bytes)
            read_end = time.perf_counter()

            if raw is None:
                stop_local_vision_decoder(proc)
                if stop_event.is_set():
                    break
                vision_decoder_restart_count += 1
                print(
                    f"[WARN] Local vision decoder frame vermedi; SADECE local decoder "
                    f"yeniden baslatiliyor (#{vision_decoder_restart_count}). "
                    "Backbone/GUI/raw kayit acik kalir."
                )
                log_system_event(
                    "vision_decoder_restart",
                    f"restart={vision_decoder_restart_count}",
                    frame_id=latest_frame_id,
                )
                time.sleep(0.10)
                proc = start_local_vision_decoder()
                last_capture_ts = None
                continue

            frame = np.frombuffer(raw, dtype=np.uint8)
            if frame.size != frame_bytes:
                continue

            frame = frame.reshape((H, W, 3)).copy()
            capture_ts = time.monotonic()
            pipe_read_ms = (read_end - read_start) * 1000.0
            capture_interval_ms = (
                0.0 if last_capture_ts is None
                else max(0.0, (capture_ts - last_capture_ts) * 1000.0)
            )
            last_capture_ts = capture_ts

            with frame_condition:
                latest_frame_id += 1
                fid = latest_frame_id
                latest_frame = frame
                latest_frame_capture_ts = capture_ts
                frame_queue.append((fid, frame))
                frame_history.append((fid, frame))
                frame_capture_times.append((fid, capture_ts))
                frame_condition.notify_all()

            note_camera_frame(fid, capture_ts)
            mode = get_active_mode()

            # Tüm modlar için hafif genel telemetri: FPS / queue / lag / delay / restart / evaluation.
            if capture_ts - last_runtime_log_ts >= 1.0:
                log_runtime_snapshot(fid)
                last_runtime_log_ts = capture_ts

            if mode == MODE_QR:
                scan_roi, scan_offset = get_target_scan_roi(frame)
                if scan_roi is not None:
                    wechat_frame_queue.put_nowait(
                        (fid, scan_roi, scan_offset, capture_ts, get_mode_generation())
                    )

            if mode != MODE_LOCK and shared_evaluation_recorder is not None:
                shared_evaluation_recorder.update(frame)

            if mode == MODE_LOCK:
                item = (fid, frame, capture_ts, pipe_read_ms, capture_interval_ms)
                while (
                    not stop_event.is_set()
                    and not ros_is_shutdown()
                    and get_active_mode() == MODE_LOCK
                ):
                    try:
                        lock_frame_queue.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue

    finally:
        stop_local_vision_decoder(proc)
        log_system_event("camera_stop", "shared_local_vision_decoder", frame_id=latest_frame_id)
        print("[INFO] Ortak vision decoder thread kapandi.")


def wait_for_latest_worker_frame(last_seen_id):
    """QR worker'a sadece QR modunda, gordugunden daha yeni EN GUNCEL frame'i verir."""
    with frame_condition:
        while not stop_event.is_set() and not ros_is_shutdown():
            if get_active_mode() == MODE_QR and latest_frame is not None and latest_frame_id > last_seen_id:
                return latest_frame_id, latest_frame, latest_frame_capture_ts
            frame_condition.wait(timeout=0.05)

        return None

def read_with_opencv_detector(detector, image, method):
    try:
        data, points, _ = detector.detectAndDecode(image)
        data = str(data).strip()
        if data and points is not None:
            return {
                "success": True,
                "readable": True,
                "text": data,
                "points": points.reshape(-1, 2).astype(np.float32),
                "method": method,
            }
    except Exception:
        pass
    return {"success": False}


def detect_candidate_with_detector(detector, image):
    try:
        detected, points = detector.detect(image)
        if detected and points is not None:
            return points.reshape(-1, 2).astype(np.float32)
    except Exception:
        pass
    return None


def create_wechat_detector_for_worker():
    if not USE_WECHAT_QR:
        print("[WECHAT WORKER] WeChatQRCode kapalı; worker çalışmayacak.")
        return None

    required = [
        WECHAT_DETECT_PROTOTXT,
        WECHAT_DETECT_CAFFEMODEL,
        WECHAT_SR_PROTOTXT,
        WECHAT_SR_CAFFEMODEL,
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        print("[WECHAT WORKER] Model dosyaları eksik; OpenCV ve pyzbar worker'ları devam edecek.")
        for path in missing:
            print(f"[WECHAT WORKER] Eksik: {path}")
        return None

    try:
        if hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            constructor = cv2.wechat_qrcode_WeChatQRCode
        elif hasattr(cv2, "wechat_qrcode") and hasattr(cv2.wechat_qrcode, "WeChatQRCode"):
            constructor = cv2.wechat_qrcode.WeChatQRCode
        else:
            print("[WECHAT WORKER] OpenCV kurulumu WeChatQRCode içermiyor.")
            return None

        detector = constructor(
            WECHAT_DETECT_PROTOTXT,
            WECHAT_DETECT_CAFFEMODEL,
            WECHAT_SR_PROTOTXT,
            WECHAT_SR_CAFFEMODEL,
        )
        print("[WECHAT WORKER] Bağımsız WeChat detector hazır.")
        return detector
    except Exception as exc:
        print(f"[WECHAT WORKER] Detector başlatılamadı: {exc}")
        return None


def read_with_wechat_detector(detector, image, method):
    if detector is None or image is None or image.size == 0:
        return {"success": False}

    try:
        decoded_info, points = detector.detectAndDecode(image)
        decoded_info = list(decoded_info) if decoded_info is not None else []
        points = list(points) if points is not None else []

        for index, text in enumerate(decoded_info):
            text = str(text).strip()
            if not text:
                continue
            pts = _normalize_wechat_points(points[index] if index < len(points) else None)
            if pts is None:
                continue
            return {
                "success": True,
                "readable": True,
                "text": text,
                "points": pts,
                "method": method,
            }
    except Exception as exc:
        now = time.monotonic()
        last_ts = getattr(read_with_wechat_detector, "last_error_ts", 0.0)
        if now - last_ts > 2.0:
            print(f"[WECHAT WORKER] Frame hatası: {exc}")
            read_with_wechat_detector.last_error_ts = now

    return {"success": False}


def read_with_pyzbar_threadsafe(image, method):
    # Ayrı pyzbar ve enhanced worker aynı anda zbar native scanner'a girmesin.
    with pyzbar_call_lock:
        result = read_with_pyzbar(image)
    if result.get("success"):
        result["readable"] = True
        result["method"] = method
    return result


def scan_full_and_center(frame, reader, full_method, center_method):
    result = reader(frame, full_method)
    if result.get("success"):
        return result

    if not USE_CENTER_SCAN_ROI:
        return {"success": False}

    center_image, offset, scale = prepare_center_scan_image(frame)
    if center_image is None:
        return {"success": False}

    result = reader(center_image, center_method)
    if result.get("success"):
        return map_result_to_full_frame(result, offset=offset, scale=scale)

    return {"success": False}


def update_worker_stats(name, elapsed_s, found):
    now = time.monotonic()
    message = None

    with worker_stats_lock:
        stat = worker_stats.setdefault(name, {
            "count": 0,
            "found": 0,
            "total_s": 0.0,
            "max_s": 0.0,
            "window_start": now,
        })
        stat["count"] += 1
        stat["found"] += int(bool(found))
        stat["total_s"] += float(elapsed_s)
        stat["max_s"] = max(stat["max_s"], float(elapsed_s))

        age = now - stat["window_start"]
        if age >= WORKER_STATS_LOG_S:
            count = max(1, stat["count"])
            rate = stat["count"] / max(age, 1e-6)
            avg_ms = stat["total_s"] * 1000.0 / count
            max_ms = stat["max_s"] * 1000.0
            message = (
                f"[PERF {name}] processed={stat['count']} rate={rate:.1f}/s "
                f"avg={avg_ms:.1f}ms max={max_ms:.1f}ms found={stat['found']}"
            )
            stat.update({
                "count": 0,
                "found": 0,
                "total_s": 0.0,
                "max_s": 0.0,
                "window_start": now,
            })

    if message:
        print(message)


def submit_worker_result(worker_name, frame_id, frame, result):
    if not result or not result.get("success"):
        return False

    submitted = result.copy()
    submitted["worker"] = worker_name
    submitted["frame_id"] = int(frame_id)
    item = (worker_name, int(frame_id), frame, submitted, time.monotonic())

    try:
        result_queue.put_nowait(item)
        return True
    except queue.Full:
        # Sonuç kuyruğu dolarsa en eski sonucu bırakıp yeni tespiti koru.
        try:
            result_queue.get_nowait()
            result_queue.task_done()
        except queue.Empty:
            pass
        try:
            result_queue.put_nowait(item)
            return True
        except queue.Full:
            return False


def detection_signature(frame_id, result):
    text = str(result.get("text", "")).strip()
    if text:
        return ("text", int(frame_id), text)

    points = result.get("points")
    if points is None:
        return ("visual", int(frame_id), result.get("method", "unknown"))

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    cx = int(np.mean(pts[:, 0]) // 16)
    cy = int(np.mean(pts[:, 1]) // 16)
    return ("visual", int(frame_id), cx, cy)


def result_dispatcher_thread_func():
    """Tüm worker sonuçlarını tek noktada tekilleştirir ve ROS/MPC'ye sıralı yollar."""
    print("[INFO] Result dispatcher thread başladı.")
    seen = {}

    while not stop_event.is_set() and not ros_is_shutdown():
        try:
            worker_name, frame_id, frame, result, submitted_ts = result_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            now = time.monotonic()
            signature = detection_signature(frame_id, result)

            # Süresi dolmuş imzaları temizle.
            expired = [key for key, ts in seen.items() if now - ts > RESULT_DEDUP_TTL_S]
            for key in expired:
                seen.pop(key, None)

            if signature in seen:
                log_detection_event(worker_name, frame_id, result, submitted_ts, "deduplicated")
                continue
            seen[signature] = now

            with frame_lock:
                current_frame_id = latest_frame_id

            lag = max(0, int(current_frame_id) - int(frame_id))
            readable = bool(str(result.get("text", "")).strip())

            # Çok eski görsel adaylar kontrol komutu üretmesin. Okunmuş QR metni yine kaybolmasın.
            if lag > RESULT_MAX_MPC_FRAME_LAG and not readable:
                log_detection_event(worker_name, frame_id, result, submitted_ts, "stale_visual_dropped", lag_frames=lag, mpc_allowed=False)
                continue

            allow_mpc = lag <= RESULT_MAX_MPC_FRAME_LAG
            result["result_lag_frames"] = lag
            result["worker"] = worker_name
            processed_result = save_and_publish_qr_result(
                result,
                frame=frame,
                frame_id=frame_id,
                allow_mpc=allow_mpc,
            )
            log_detection_event(
                worker_name, frame_id, result, submitted_ts, "processed",
                lag_frames=lag, mpc_allowed=allow_mpc, processed_result=processed_result,
            )
        finally:
            result_queue.task_done()

    print("[INFO] Result dispatcher thread kapandı.")


def _latest_frame_id_snapshot():
    with frame_lock:
        return int(latest_frame_id)


def opencv_worker_thread_func():
    detector = cv2.QRCodeDetector()
    last_seen_id = 0
    every_n = max(1, OPENCV_WORKER_EVERY_N)
    profile = "opencv detectAndDecode: full frame + center ROI"
    print("[INFO] OpenCV worker başladı: tam frame + merkez ROI, her uygun kare.")

    while not stop_event.is_set() and not ros_is_shutdown():
        item = wait_for_latest_worker_frame(last_seen_id)
        if item is None:
            continue
        frame_id, frame, capture_ts = item
        forced_gap = max(0, frame_id - last_seen_id - 1)
        last_seen_id = frame_id
        latest_start = _latest_frame_id_snapshot()

        if frame_id % every_n != 0:
            if LOG_SCHEDULED_SKIPS:
                now = time.monotonic()
                log_worker_attempt("opencv", "scheduled_skip", frame_id, latest_start,
                                   _latest_frame_id_snapshot(), forced_gap, every_n,
                                   capture_ts, now, now, attempt_profile=profile)
            continue

        started_mono = time.monotonic()
        started_perf = time.perf_counter()
        result = scan_full_and_center(
            frame,
            lambda image, method: read_with_opencv_detector(detector, image, method),
            "parallel_opencv_full",
            "parallel_opencv_center",
        )
        result["capture_monotonic"] = capture_ts
        submitted = submit_worker_result("opencv", frame_id, frame, result)
        elapsed = time.perf_counter() - started_perf
        finished_mono = time.monotonic()
        update_worker_stats("opencv", elapsed, submitted)
        log_worker_attempt("opencv", "processed", frame_id, latest_start,
                           _latest_frame_id_snapshot(), forced_gap, every_n,
                           capture_ts, started_mono, finished_mono, result,
                           submitted, profile)

    print("[INFO] OpenCV worker kapandı.")



def wechat_worker_thread_func():
    """Q modunda HER kamera karesini FIFO sirayla, vurus alani + margin ROI'de TEK KEZ WeChat ile tarar."""
    global wechat_forced_skip_total, wechat_last_q_warn_ts

    detector = create_wechat_detector_for_worker()
    if detector is None:
        log_system_event("worker_disabled", "WeChat detector başlatılamadı")
        return

    last_seen_id = 0
    every_n = 1
    profile = "WeChatQRCode: target ROI + margin, FIFO every frame, single pass"
    print(f"[INFO] WeChat worker: HER FRAME FIFO | TEK OKUMA | vurus alani + {TARGET_SCAN_MARGIN_PX}px margin.")

    while not stop_event.is_set() and not ros_is_shutdown():
        try:
            frame_id, scan_roi, offset, capture_ts, worker_mode_generation = wechat_frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            if get_active_mode() != MODE_QR or int(worker_mode_generation) != get_mode_generation():
                continue

            forced_gap = max(0, int(frame_id) - int(last_seen_id) - 1) if last_seen_id else 0
            if forced_gap:
                with live_perf_lock:
                    wechat_forced_skip_total += forced_gap
            last_seen_id = int(frame_id)
            latest_start = _latest_frame_id_snapshot()

            started_mono = time.monotonic()
            started_perf = time.perf_counter()

            result = read_with_wechat_detector(detector, scan_roi, "parallel_wechat_target_roi")
            if result.get("success"):
                result = map_result_to_full_frame(result, offset=offset, scale=1.0)

            result["capture_monotonic"] = capture_ts
            result["mode_generation"] = int(worker_mode_generation)
            submitted = submit_worker_result("wechat", frame_id, None, result)

            elapsed = time.perf_counter() - started_perf
            finished_mono = time.monotonic()
            note_wechat_processed(frame_id, finished_mono, capture_ts)
            update_worker_stats("wechat", elapsed, submitted)
            log_worker_attempt(
                "wechat", "processed", frame_id, latest_start,
                _latest_frame_id_snapshot(), forced_gap, every_n,
                capture_ts, started_mono, finished_mono, result,
                submitted, profile
            )

            q_size = wechat_frame_queue.qsize()
            if q_size >= WECHAT_QUEUE_WARN_SIZE and finished_mono - wechat_last_q_warn_ts >= 2.0:
                print(
                    f"[WARN] WeChat FIFO Q={q_size}. WeChat kamera FPS'ine yetisemiyor; "
                    "frame atilmiyor fakat gecikme birikiyor."
                )
                wechat_last_q_warn_ts = finished_mono
        finally:
            wechat_frame_queue.task_done()

    print("[INFO] WeChat worker kapandi.")



def pyzbar_worker_thread_func():
    """Pyzbar fallback her 5. uygun frame'de vurus alani + margin ROI tarar."""
    last_seen_id = 0
    every_n = max(1, PYZBAR_WORKER_EVERY_N)
    profile = "pyzbar/zbar: target ROI + margin"
    print(f"[INFO] pyzbar worker: her {every_n} frame | vurus alani + {TARGET_SCAN_MARGIN_PX}px margin.")

    while not stop_event.is_set() and not ros_is_shutdown():
        item = wait_for_latest_worker_frame(last_seen_id)
        if item is None:
            continue

        frame_id, frame, capture_ts = item
        worker_mode_generation = get_mode_generation()
        forced_gap = max(0, frame_id - last_seen_id - 1)
        last_seen_id = frame_id
        latest_start = _latest_frame_id_snapshot()

        if frame_id % every_n != 0:
            if LOG_SCHEDULED_SKIPS:
                now = time.monotonic()
                log_worker_attempt(
                    "pyzbar", "scheduled_skip", frame_id, latest_start,
                    _latest_frame_id_snapshot(), forced_gap, every_n,
                    capture_ts, now, now, attempt_profile=profile
                )
            continue

        scan_roi, offset = get_target_scan_roi(frame)
        if scan_roi is None:
            continue

        started_mono = time.monotonic()
        started_perf = time.perf_counter()
        result = read_with_pyzbar_threadsafe(scan_roi, "parallel_pyzbar_target_roi")
        if result.get("success"):
            result = map_result_to_full_frame(result, offset=offset, scale=1.0)

        result["capture_monotonic"] = capture_ts
        result["mode_generation"] = int(worker_mode_generation)
        submitted = submit_worker_result("pyzbar", frame_id, frame, result)
        elapsed = time.perf_counter() - started_perf
        finished_mono = time.monotonic()
        update_worker_stats("pyzbar", elapsed, submitted)
        log_worker_attempt(
            "pyzbar", "processed", frame_id, latest_start,
            _latest_frame_id_snapshot(), forced_gap, every_n,
            capture_ts, started_mono, finished_mono, result,
            submitted, profile
        )

    print("[INFO] pyzbar worker kapandi.")


def add_offset_to_result(result, offset, method):
    if not result.get("success"):
        return result
    mapped = result.copy()
    points = mapped.get("points")
    if points is not None:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        pts[:, 0] += float(offset[0])
        pts[:, 1] += float(offset[1])
        mapped["points"] = pts
    mapped["method"] = method
    return mapped


def enhanced_candidate_pipeline(frame, detector, method_prefix="parallel_enhanced"):
    """Aday köşeyi bulur; ROI iyileştirme, perspektif ve unreadable visual target üretir."""
    candidate_points = detect_candidate_with_detector(detector, frame)

    if candidate_points is None and USE_CENTER_SCAN_ROI:
        center_image, offset, scale = prepare_center_scan_image(frame)
        if center_image is not None:
            center_points = detect_candidate_with_detector(detector, center_image)
            if center_points is not None:
                candidate_points = np.asarray(center_points, dtype=np.float32).copy()
                candidate_points[:, 0] = candidate_points[:, 0] / scale + offset[0]
                candidate_points[:, 1] = candidate_points[:, 1] / scale + offset[1]

    if candidate_points is None:
        return {"success": False}

    roi, offset = extract_roi(frame, candidate_points, margin=ROI_MARGIN)
    if roi is None:
        return {"success": False}

    ox, oy = offset
    local_points = np.asarray(candidate_points, dtype=np.float32).copy()
    local_points[:, 0] -= ox
    local_points[:, 1] -= oy

    variants = [
        ("raw", roi),
        ("clahe", apply_clahe_roi(roi)),
        ("threshold", apply_adaptive_threshold_roi(roi)),
        ("sharpen", apply_unsharp_mask(roi)),
    ]

    for variant_name, variant in variants:
        result = read_with_opencv_detector(
            detector,
            variant,
            f"{method_prefix}_{variant_name}_opencv",
        )
        if result.get("success"):
            return add_offset_to_result(result, offset, result["method"])

        result = read_with_pyzbar_threadsafe(
            variant,
            f"{method_prefix}_{variant_name}_pyzbar",
        )
        if result.get("success"):
            return add_offset_to_result(result, offset, result["method"])

    threshold_roi = variants[2][1]
    clahe_roi = variants[1][1]
    quad = find_quad_by_contours(threshold_roi)
    if quad is None:
        quad = find_quad_by_contours(clahe_roi)
    if quad is None:
        quad = order_points(local_points)
    if quad is None:
        return {"success": False}

    warped = perspective_warp(roi, quad, size=PERSPECTIVE_SIZE)
    if warped is not None:
        result = read_with_opencv_detector(
            detector,
            warped,
            f"{method_prefix}_warp_opencv",
        )
        if result.get("success"):
            full_points = np.asarray(quad, dtype=np.float32).copy()
            full_points[:, 0] += ox
            full_points[:, 1] += oy
            result["points"] = full_points
            return result

        result = read_with_pyzbar_threadsafe(
            warped,
            f"{method_prefix}_warp_pyzbar",
        )
        if result.get("success"):
            full_points = np.asarray(quad, dtype=np.float32).copy()
            full_points[:, 0] += ox
            full_points[:, 1] += oy
            result["points"] = full_points
            return result

    unreadable_points = np.asarray(quad, dtype=np.float32).copy()
    unreadable_points[:, 0] += ox
    unreadable_points[:, 1] += oy
    return {
        "success": True,
        "readable": False,
        "text": "",
        "points": unreadable_points,
        "method": f"{method_prefix}_unreadable_visual_candidate",
    }


def enhanced_variant_decode(frame, detector, method_prefix="parallel_enhanced_periodic"):
    scan_image, offset, scale = prepare_center_scan_image(frame)
    if scan_image is None:
        return {"success": False}

    variants = [
        ("raw", scan_image),
        ("clahe", apply_clahe_roi(scan_image)),
        ("threshold", apply_adaptive_threshold_roi(scan_image)),
        ("sharpen", apply_unsharp_mask(scan_image)),
    ]

    for variant_name, variant in variants:
        result = read_with_opencv_detector(
            detector,
            variant,
            f"{method_prefix}_{variant_name}_opencv",
        )
        if result.get("success"):
            return map_result_to_full_frame(result, offset=offset, scale=scale)

        result = read_with_pyzbar_threadsafe(
            variant,
            f"{method_prefix}_{variant_name}_pyzbar",
        )
        if result.get("success"):
            return map_result_to_full_frame(result, offset=offset, scale=scale)

    return {"success": False}


def enhanced_worker_thread_func():
    detector = cv2.QRCodeDetector()
    last_seen_id = 0
    every_n = max(1, ENHANCED_WORKER_EVERY_N)
    profile = "candidate ROI + CLAHE + threshold + sharpen + perspective + history"
    print("[INFO] Enhanced worker başladı: aday/ROI/CLAHE/threshold/warp + geçmiş keskin kareler.")

    while not stop_event.is_set() and not ros_is_shutdown():
        item = wait_for_latest_worker_frame(last_seen_id)
        if item is None:
            continue
        frame_id, frame, capture_ts = item
        forced_gap = max(0, frame_id - last_seen_id - 1)
        last_seen_id = frame_id
        latest_start = _latest_frame_id_snapshot()

        if frame_id % every_n != 0:
            if LOG_SCHEDULED_SKIPS:
                now = time.monotonic()
                log_worker_attempt("enhanced", "scheduled_skip", frame_id, latest_start,
                                   _latest_frame_id_snapshot(), forced_gap, every_n,
                                   capture_ts, now, now, attempt_profile=profile)
            continue

        started_mono = time.monotonic()
        started_perf = time.perf_counter()
        result = enhanced_candidate_pipeline(frame, detector)
        if not result.get("success"):
            result = enhanced_variant_decode(frame, detector)

        result["capture_monotonic"] = capture_ts
        submitted = submit_worker_result("enhanced", frame_id, frame, result)
        found = submitted

        if not found and PERIODIC_RECENT_TOP_K > 0:
            recent_frames = get_sharpest_recent_frames(
                limit=PERIODIC_RECENT_SCAN_FRAMES,
                top_k=PERIODIC_RECENT_TOP_K,
                exclude_frame_id=frame_id,
            )
            for recent_id, recent_frame in recent_frames:
                if stop_event.is_set() or ros_is_shutdown():
                    break
                historical_capture_ts = get_frame_capture_ts(recent_id)
                hist_start = time.monotonic()
                historical = enhanced_variant_decode(
                    recent_frame,
                    detector,
                    method_prefix="parallel_enhanced_history",
                )
                historical["capture_monotonic"] = historical_capture_ts
                hist_submitted = submit_worker_result("enhanced_history", recent_id, recent_frame, historical)
                hist_finish = time.monotonic()
                log_worker_attempt(
                    "enhanced_history", "processed", recent_id, latest_start,
                    _latest_frame_id_snapshot(), 0, 1, historical_capture_ts,
                    hist_start, hist_finish, historical, hist_submitted,
                    "historical sharp frame: CLAHE + threshold + sharpen",
                )
                if hist_submitted:
                    found = True
                    break

        elapsed = time.perf_counter() - started_perf
        finished_mono = time.monotonic()
        update_worker_stats("enhanced", elapsed, found)
        log_worker_attempt("enhanced", "processed", frame_id, latest_start,
                           _latest_frame_id_snapshot(), forced_gap, every_n,
                           capture_ts, started_mono, finished_mono, result,
                           submitted, profile)

    print("[INFO] Enhanced worker kapandı.")


def get_latest_frame_for_display():
    with frame_lock:
        if latest_frame is None:
            return None, 0
        return latest_frame.copy(), latest_frame_id


def get_last_qr_for_display():
    with qr_result_lock:
        if last_qr_result is None:
            return None, 0.0
        return last_qr_result.copy(), last_qr_ts


# ============================================================================
# TUNGA COMBINED - PERMANENT VIDEO BACKBONE + Q/K MODE SWITCH
# ============================================================================
# Bu bolum QR ve LOCK algoritmalarini birbirine karistirmadan tek process icinde
# yonetir. Herelink'e sadece PersistentVideoBackbone baglanir.
#
# TUSLAR:
#   Q/q : QR modu (WeChat her frame, pyzbar her 5 frame)
#   K/k : Kilitlenme modu (her gelen LOCK frame -> YOLO, app-level drop yok)
#   Z/z : QR ekrani yazilimsal zoom
#   ESC : tum sistemi duzgun kapat
#
# Q/K gecisleri BACKBONE FFMPEG'i, GUI UDP'yi, raw MKV'yi veya evaluation
# recorder'i ASLA stop/start etmez.
# ============================================================================

# Lock kodundan gelen sabitler (mevcut degerler aynen korunur).
DEFAULT_TARGET_DATA_TOPIC = "/target_data"
DEFAULT_HEDEF_PIKSEL_TOPIC = "/hedef_piksel"
DEFAULT_LOCK_TOPIC = "/kilitlenme_bilgisi"
DEFAULT_SERVER_TIME_TOPIC = "/server_time"
DEFAULT_LOCK_DURATION_S = 4.0
DEFAULT_LOCK_LOSS_TOLERANCE_S = 0.200
DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/fpv_stream"
DEFAULT_PROCESS_WIDTH = 1280
DEFAULT_PROCESS_HEIGHT = 720
DEFAULT_CAPTURE_FPS = 30

# Goruntunun EN BASINDA ustten kirpilacak piksel sayisi.
# Kirpma local vision decoder'da scale'den once uygulanir; QR/YOLO/ByteTrack/
# evaluation/display taraflari kirpilmis 1280x720 frame'i gorur.
# 0 yapilirsa kirpma kapanir.
DEFAULT_TOP_CROP_PX = 36
TOP_CROP_PX = DEFAULT_TOP_CROP_PX
DEFAULT_GUI_STREAM_HOST = "127.0.0.1"
DEFAULT_GUI_STREAM_PORT = 5000
DEFAULT_GUI_PKT_SIZE = 1316
DEFAULT_RAW_RECORD_ENABLED = True
DEFAULT_GUI_STREAM_ENABLED = True
DEFAULT_EVALUATION_RECORD_ENABLED = True
DEFAULT_EVALUATION_FPS = 30.0
DEFAULT_EVALUATION_ENCODER = "libx264"
DEFAULT_EVALUATION_CRF = 18
DEFAULT_EVALUATION_PRESET = "ultrafast"
DEFAULT_READER_QUEUE_SIZE = 5

# Sartnameye gore degerlendirme videosu minimum 640x480 olmali.
MIN_COMPETITION_WIDTH = 640
MIN_COMPETITION_HEIGHT = 480
ALLOWED_COMPETITION_ASPECT_RATIOS = (4.0 / 3.0, 5.0 / 4.0, 16.0 / 9.0)
ASPECT_RATIO_TOLERANCE = 0.01

# Müsabaka videosu dosya adi icin sabit takim adi (ASCII / Turkce karakter yok).
COMPETITION_TEAM_FILE_NAME = "TUNGA_SAYE_IHA_Takimi"
DEFAULT_SERVER_TIME_STARTUP_TIMEOUT_S = 10.0

DEFAULT_HIT_LEFT_RATIO = 0.25
DEFAULT_HIT_RIGHT_RATIO = 0.75
DEFAULT_HIT_TOP_RATIO = 0.10
DEFAULT_HIT_BOTTOM_RATIO = 0.90
DEFAULT_MIN_TARGET_AXIS_RATIO = 0.06

VALID_BOX_COLOR_BGR = (0, 0, 255)
INVALID_BOX_COLOR_BGR = (0, 255, 255)
SELECTED_BOX_COLOR_BGR = (0, 0, 255)
HIT_AREA_COLOR_BGR = (0, 255, 255)
VALID_CENTER_COLOR_BGR = (0, 0, 255)
INVALID_CENTER_COLOR_BGR = (0, 255, 255)
# K/ByteTrack manuel hedef secimi renkleri
MANUAL_OTHER_COLOR_BGR = (0, 0, 0)          # Siyah: diger ByteTrack hedefleri
MANUAL_SELECTED_COLOR_BGR = (255, 255, 0)   # Camgobegi: SPACE bekleyen secili hedef
MANUAL_FOCUSED_COLOR_BGR = (0, 255, 0)      # Yesil: SPACE acik + ByteTrack aktif
MANUAL_FALLBACK_COLOR_BGR = (255, 0, 255)   # Mor: ByteTrack yok, raw YOLO fallback
FPS_TEXT_COLOR_BGR = (0, 0, 0)
CLOCK_TEXT_COLOR_BGR = (255, 255, 255)
CLOCK_OUTLINE_COLOR_BGR = (0, 0, 0)
DETECTION_BOX_THICKNESS = 2
SELECTED_BOX_THICKNESS = 3
HIT_AREA_THICKNESS = 2

COMBINED_WINDOW_TITLE = "TUNGA SINGLE RTSP - Q:QR / K:LOCK / ESC:EXIT"
DEFAULT_VISION_RELAY_HOST = "127.0.0.1"
DEFAULT_VISION_RELAY_PORT = 5601

# Tek ROS import ismini iki eski kod icin ortak kullan.
RosString = String

MODE_IDLE = "IDLE"
MODE_QR = "QR"
MODE_LOCK = "LOCK"
_mode_lock = threading.Lock()
active_mode = MODE_IDLE
mode_generation = 0

lock_frame_queue = queue.Queue(maxsize=DEFAULT_READER_QUEUE_SIZE)
lock_display_lock = threading.Lock()
latest_lock_display_frame = None
lock_engine_instance = None
shared_evaluation_recorder = None
shared_server_time_provider = None
shared_backbone = None
vision_decoder_restart_count = 0
vision_relay_host = DEFAULT_VISION_RELAY_HOST
vision_relay_port = DEFAULT_VISION_RELAY_PORT


def prompt_competition_number():
    """Program basinda sartnameye uygun video adi icin musabaka numarasini ister."""
    while True:
        try:
            value = input("Musabaka numarasini girin: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError("Musabaka numarasi girilmeden program baslatilamaz.")

        if value.isdigit() and int(value) > 0:
            return str(int(value))

        print("[HATA] Musabaka numarasi pozitif bir tam sayi olmali. Ornek: 2")


def build_competition_video_filename(competition_number):
    date_text = datetime.now().strftime("%d_%m_%Y")
    return f"{competition_number}_{COMPETITION_TEAM_FILE_NAME}_{date_text}.mp4"


def validate_competition_resolution(width, height):
    """Minimum cozunurluk ve izin verilen aspect ratio sartlarini zorunlu tutar."""
    width = int(width)
    height = int(height)
    if width < MIN_COMPETITION_WIDTH or height < MIN_COMPETITION_HEIGHT:
        raise RuntimeError(
            f"Degerlendirme/processing cozunurlugu en az "
            f"{MIN_COMPETITION_WIDTH}x{MIN_COMPETITION_HEIGHT} olmali; "
            f"mevcut={width}x{height}."
        )

    ratio = width / float(height)
    if not any(abs(ratio - allowed) <= ASPECT_RATIO_TOLERANCE for allowed in ALLOWED_COMPETITION_ASPECT_RATIOS):
        raise RuntimeError(
            f"Gecersiz aspect ratio: {width}x{height} ({ratio:.4f}). "
            "Sartnameye gore yalnizca 4:3, 5:4 veya 16:9 kullanilabilir."
        )


def get_active_mode():
    # CPython'da string referans okumasi atomiktir; worker hot-path'inde ekstra lock yok.
    return active_mode


def get_mode_generation():
    return mode_generation


def _drain_queue(q):
    while True:
        try:
            q.get_nowait()
            try:
                q.task_done()
            except Exception:
                pass
        except queue.Empty:
            break



def _clear_qr_transient_state():
    global last_qr_result, last_qr_ts, last_qr_frame_id, last_publish_ts
    global qr_package_sent
    _drain_queue(result_queue)
    _drain_queue(wechat_frame_queue)
    with qr_result_lock:
        last_qr_result = None
        last_qr_ts = 0.0
        last_qr_frame_id = -1
    last_publish_ts = 0.0
    with qr_package_sent_lock:
        qr_package_sent = False


def set_active_mode(new_mode):
    """Q/K mode switch. Video backbone ve kayit proseslerine DOKUNMAZ."""
    global active_mode, mode_generation, latest_lock_display_frame

    new_mode = str(new_mode).upper().strip()
    if new_mode not in (MODE_IDLE, MODE_QR, MODE_LOCK):
        return False

    with _mode_lock:
        old_mode = active_mode
        if old_mode == new_mode:
            return False
        active_mode = new_mode
        mode_generation += 1
        gen = mode_generation

    # Eski modun bekleyen islerini temizle. Backbone/decoder kapanmaz.
    if new_mode != MODE_LOCK:
        _drain_queue(lock_frame_queue)
        with lock_display_lock:
            latest_lock_display_frame = None

    if new_mode != MODE_QR:
        _clear_qr_transient_state()

    if new_mode == MODE_QR:
        # Q'ya girerken eski QR overlay/yayin kalintisini da sifirla.
        _clear_qr_transient_state()

    # Condition'da uyuyan QR worker'larini mode degisikligi icin uyandir.
    with frame_condition:
        frame_condition.notify_all()

    log_system_event("mode_change", f"{old_mode}->{new_mode}; generation={gen}", frame_id=latest_frame_id)
    print(f"\n[MODE] {old_mode} -> {new_mode}")
    if new_mode == MODE_QR:
        print(f"[MODE] QR AKTIF: WeChat HER FRAME FIFO/TEK OKUMA, vurus alani + {TARGET_SCAN_MARGIN_PX}px; pyzbar her 5 frame. YOLO DURUR.")
    elif new_mode == MODE_LOCK:
        print("[MODE] LOCK AKTIF: Her K frame 1x YOLO+ByteTrack; ayni track ID lock. QR worker'lari UYUR.")
    else:
        print("[MODE] IDLE: Goruntu/yayin/kayit devam; QR ve YOLO inference yok.")
    return True


def _part_path(base_path, part_index):
    base = Path(base_path)
    if part_index <= 1:
        return base
    return base.with_name(f"{base.stem}_part{part_index:02d}{base.suffix}")


class PersistentVideoBackbone:
    """
    Ucus boyunca yasayan TEK Herelink RTSP istemcisi.

    Tek RTSP input -> H264 stream-copy tee:
      1) raw MKV
      2) GUI UDP
      3) localhost vision relay

    Her tee slave FIFO ile izole edilir. Vision decoder/YOLO/QR bloke olsa bile Python
    mode switch'i bu prosesi stop/start etmez.
    """

    def __init__(
        self,
        rtsp_url,
        raw_base_path,
        gui_enabled,
        gui_host,
        gui_port,
        relay_host,
        relay_port,
        ffmpeg_log_path,
    ):
        self.rtsp_url = str(rtsp_url)
        self.raw_base_path = Path(raw_base_path) if raw_base_path else None
        self.gui_enabled = bool(gui_enabled)
        self.gui_host = str(gui_host)
        self.gui_port = int(gui_port)
        self.relay_host = str(relay_host)
        self.relay_port = int(relay_port)
        self.ffmpeg_log_path = Path(ffmpeg_log_path)
        self.process = None
        self.stderr_handle = None
        self.start_count = 0
        self.restart_count = 0
        self.raw_paths = []
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self._process_lock = threading.Lock()

    def _open(self):
        self.start_count += 1
        part_index = self.start_count
        raw_path = None
        if self.raw_base_path is not None:
            raw_path = _part_path(self.raw_base_path, part_index)
            self.raw_paths.append(raw_path)

        targets = []
        if raw_path is not None:
            targets.append(f"[onfail=ignore:f=matroska]{raw_path}")

        if self.gui_enabled:
            gui_url = f"udp://{self.gui_host}:{self.gui_port}?pkt_size={DEFAULT_GUI_PKT_SIZE}"
            targets.append(f"[onfail=ignore:f=mpegts]{gui_url}")

        # Local vision kolu HER ZAMAN acik. Bu ikinci Herelink RTSP degildir.
        local_url = f"udp://{self.relay_host}:{self.relay_port}?pkt_size={DEFAULT_GUI_PKT_SIZE}"
        targets.append(f"[onfail=ignore:f=mpegts]{local_url}")

        self.ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_handle = open(self.ffmpeg_log_path, "ab", buffering=0)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "genpts+igndts+discardcorrupt",
            "-flags", "low_delay",
            "-rtsp_transport", "udp",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-stimeout", "2000000",
            "-i", self.rtsp_url,
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-f", "tee",
            "-use_fifo", "1",
            "-fifo_options", "attempt_recovery=1:recover_any_error=1",
            "|".join(targets),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr_handle,
            bufsize=0,
        )

        print("[BACKBONE] Herelink RTSP acildi. CLIENT SAYISI = 1")
        if raw_path is not None:
            print(f"[BACKBONE] Raw kayit: {raw_path}")
        if self.gui_enabled:
            print(f"[BACKBONE] GUI: udp://{self.gui_host}:{self.gui_port} (H264 copy)")
        print(f"[BACKBONE] Local vision relay: udp://{self.relay_host}:{self.relay_port} (H264 copy)")

    def start(self):
        self.stop_event.clear()
        with self._process_lock:
            self._close_process(graceful=False)
            self._open()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="video_backbone_monitor",
            daemon=True,
        )
        self.monitor_thread.start()

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            time.sleep(0.20)
            with self._process_lock:
                proc = self.process
                dead = proc is None or proc.poll() is not None
            if not dead:
                continue
            if self.stop_event.is_set():
                break
            self.restart_count += 1
            print(
                f"[BACKBONE WARN] RTSP backbone beklenmedik kapandi; yeniden baglaniliyor "
                f"(restart #{self.restart_count})."
            )
            log_system_event(
                "backbone_restart",
                f"restart={self.restart_count}",
                frame_id=latest_frame_id,
                value=self.restart_count,
            )
            try:
                with self._process_lock:
                    self._close_process(graceful=False)
                    self._open()
            except Exception as exc:
                print(f"[BACKBONE ERROR] Yeniden baslatma hatasi: {exc}")
                time.sleep(0.5)

    def _close_process(self, graceful=True):
        proc = self.process
        self.process = None
        if proc is not None:
            try:
                if proc.poll() is None and graceful:
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=4.0)
                elif proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        if self.stderr_handle is not None:
            try:
                self.stderr_handle.close()
            except Exception:
                pass
            self.stderr_handle = None

    def stop(self):
        self.stop_event.set()
        with self._process_lock:
            self._close_process(graceful=True)
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None


def start_local_vision_decoder():
    """Sadece localhost relay'i decode eder; Herelink'e BAGLANMAZ."""
    url = (
        f"udp://{vision_relay_host}:{vision_relay_port}"
        "?fifo_size=1000000&overrun_nonfatal=1&timeout=2000000"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-fflags", "nobuffer+genpts+igndts+discardcorrupt",
        "-flags", "low_delay",
        "-analyzeduration", "1000000",
        "-probesize", "1000000",
        "-i", url,
        "-map", "0:v:0",
        "-an",
        # EN ILK goruntu islemi: ustteki istenmeyen bandi kirp, SONRA
        # sartnameye uygun sabit processing boyutuna getir. Boylece Python'a giren
        # tum QR/YOLO/ByteTrack/evaluation/display islemleri kirpma SONRASINDA calisir.
        # Nihai frame boyutu yine W x H'dir (varsayilan 1280x720, 16:9).
        "-vf", (
            f"crop=iw:ih-{int(TOP_CROP_PX)}:0:{int(TOP_CROP_PX)},"
            f"scale={int(W)}:{int(H)}"
            if int(TOP_CROP_PX) > 0
            else f"scale={int(W)}:{int(H)}"
        ),
        "-pix_fmt", "bgr24",
        "-c:v", "rawvideo",
        "-f", "rawvideo",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )


def _read_exactly_from_proc(proc, nbytes):
    if proc is None or proc.stdout is None:
        return None
    chunks = bytearray()
    while len(chunks) < nbytes and not stop_event.is_set():
        piece = proc.stdout.read(nbytes - len(chunks))
        if not piece:
            return None
        chunks.extend(piece)
    if len(chunks) != nbytes:
        return None
    return bytes(chunks)


def stop_local_vision_decoder(proc):
    if proc is None:
        return
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:
        pass
    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=1.5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass



# ---------------- LOCK KODUNDAN AYNI ALGORITMA FONKSIYONLARI ----------------

def publish_json(ros_active, publisher, topic_name, payload, terminal_log=True):
    encoded = json.dumps(payload, ensure_ascii=False)

    if ros_active and publisher is not None:
        publisher.publish(RosString(encoded))
    elif terminal_log:
        print(f"[{topic_name}] {encoded}")


def publish_lock_success(ros_active, lock_pub, lock_topic, finish_time, target_id=None):
    payload = {
        "kilitlenmeBitisZamani": {
            "saat": finish_time.hour,
            "dakika": finish_time.minute,
            "saniye": finish_time.second,
            "milisaniye": finish_time.microsecond // 1000,
        },
        "otonom_kilitlenme": 1,
    }
    publish_json(ros_active, lock_pub, lock_topic, payload, terminal_log=True)
    if target_id is None:
        print("[LOCK] KILITLENME BASARILI")
    else:
        print(f"[LOCK] KILITLENME BASARILI | target_id={target_id}")


class ServerTimeProvider:
    """
    Yarisma degerlendirme videosunda SADECE /server_time kullanilir.

    Local PC clock fallback YOKTUR. /server_time ilk gecerli mesaji gelmeden
    sistem baslatilmaz. Callback hem dogrudan:
        {"saat": 12, "dakika": 34, "saniye": 56, "milisaniye": 789}
    hem de bu alanlari iceren nested JSON yapilarini destekler.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._server_dt = None
        self._server_rx_monotonic = None
        self._source = "WAITING_FOR_SERVER_TIME"
        self._ready_event = threading.Event()

    def is_ready(self):
        return self._ready_event.is_set()

    def wait_until_ready(self, timeout_s):
        return self._ready_event.wait(timeout=max(0.0, float(timeout_s)))

    def now(self):
        with self._lock:
            server_dt = self._server_dt
            rx_mono = self._server_rx_monotonic

        if server_dt is None or rx_mono is None:
            raise RuntimeError(
                "/server_time alinmadan sunucu saati kullanilamaz; "
                "local PC clock fallback devre disidir."
            )

        # Son topic mesajindan beri gecen sureyi ekleyerek saat donmasini onle.
        elapsed = max(0.0, time.monotonic() - rx_mono)
        return server_dt + timedelta(seconds=elapsed)

    def source_name(self):
        with self._lock:
            return self._source

    def server_time_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            candidate = payload

            # Bazi olasi nested alan adlari icin toleransli arama.
            for key in ("serverTime", "server_time", "sunucuSaati", "sunucu_saati"):
                if isinstance(payload, dict) and isinstance(payload.get(key), dict):
                    candidate = payload[key]
                    break

            hour = int(candidate["saat"])
            minute = int(candidate["dakika"])
            second = int(candidate["saniye"])
            millisecond = int(candidate.get("milisaniye", 0))

            local_now = datetime.now()
            server_dt = local_now.replace(
                hour=hour,
                minute=minute,
                second=second,
                microsecond=max(0, min(999, millisecond)) * 1000,
            )

            with self._lock:
                self._server_dt = server_dt
                self._server_rx_monotonic = time.monotonic()
                self._source = DEFAULT_SERVER_TIME_TOPIC
            self._ready_event.set()

        except Exception as exc:
            print(f"[WARN] /server_time parse edilemedi: {exc}")


def format_server_time(dt):
    return f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{dt.microsecond // 1000:03d}"


def draw_server_time(frame, server_dt, anchor="right", y=32):
    text = f"SERVER TIME {format_server_time(server_dt)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.67
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)

    if anchor == "center":
        x = max(10, (frame.shape[1] - text_w) // 2)
    elif anchor == "left":
        x = 15
    else:
        x = max(10, frame.shape[1] - text_w - 15)

    # Okunabilirlik icin sadece saatin kendi outline'i.
    cv2.putText(
        frame, text, (x, y), font, scale,
        CLOCK_OUTLINE_COLOR_BGR, 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x, y), font, scale,
        CLOCK_TEXT_COLOR_BGR, thickness, cv2.LINE_AA,
    )


def build_evaluation_base_frame(clean_frame, lock_candidate):
    """
    Musabaka sonrasi kayit icin SADECE timer'in kullandigi TEK lock candidate'i
    KIRMIZI kutu ile cizer. Diger detection'lar evaluation videosuna cizilmez.

    YOK:
      - vurus alani dikdortgeni
      - diger/sari kutular
      - confidence
      - center noktasi
      - FPS
      - Q
      - kilit timer
      - tolerans yazisi

    Saat burada cizilmez; EvaluationRecorder sabit FPS thread'i her output frame'inde
    en guncel server/local saati cizer.
    """
    out = clean_frame.copy()
    if lock_candidate is not None:
        x1, y1, x2, y2 = lock_candidate["xyxy_int"]
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            VALID_BOX_COLOR_BGR,
            SELECTED_BOX_THICKNESS,
        )
    return out


class EvaluationRecorder:
    """
    Musabaka sonrasi video kaydini YOLO loop'undan ayirir.

    - main loop sadece en yeni temiz evaluation frame'ini gunceller.
    - ayri thread SABIT FPS ile en yeni frame'i FFmpeg'e yazar.
    - YOLO yavaslarsa son frame tekrar edilir; recording FPS sabit kalir.
    - her output frame'inde saat yeniden cizilir.
    - encoder yavaslarsa YOLO thread'i beklemez; gecikme recorder thread'inde kalir.
    """

    def __init__(
        self,
        output_path,
        width,
        height,
        fps,
        time_provider,
        ffmpeg_log_path,
        encoder=DEFAULT_EVALUATION_ENCODER,
        preset=DEFAULT_EVALUATION_PRESET,
        crf=DEFAULT_EVALUATION_CRF,
    ):
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self.time_provider = time_provider
        self.ffmpeg_log_path = Path(ffmpeg_log_path)
        self.encoder = str(encoder)
        self.preset = str(preset)
        self.crf = int(crf)

        self.process = None
        self.thread = None
        self.stop_event = threading.Event()
        self._stderr_handle = None

        self._frame_lock = threading.Lock()
        self._latest_base_frame = None
        self._latest_version = 0

        self.total_written_frames = 0
        self.duplicate_frames = 0
        self.deadline_misses = 0
        self.broken_pipe = False
        self.write_ms_samples = []

    def _build_cmd(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", f"{self.fps:.6f}",
            "-i", "pipe:0",
            "-an",
        ]

        if self.encoder == "h264_nvenc":
            # NVIDIA varsa config.yaml: evaluation_encoder: h264_nvenc kullanilabilir.
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-tune", "ll",
                "-rc", "vbr",
                "-cq", str(self.crf),
            ]
        else:
            cmd += [
                "-c:v", self.encoder,
                "-preset", self.preset,
                "-tune", "zerolatency",
                "-crf", str(self.crf),
            ]

        cmd += [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.output_path),
        ]
        return cmd

    def start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = open(self.ffmpeg_log_path, "ab", buffering=0)
        self.process = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_handle,
            bufsize=0,
        )
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._writer_loop,
            name="evaluation_cfr_recorder",
            daemon=True,
        )
        self.thread.start()
        print(
            f"[INFO] Evaluation recorder: {self.output_path} "
            f"({self.width}x{self.height} @ {self.fps:.1f} CFR, {self.encoder})"
        )

    def update(self, base_frame):
        start = time.perf_counter()
        # Kopya main loop'taki debug cizimlerinden evaluation kaydini tamamen izole eder.
        safe_frame = base_frame.copy()
        with self._frame_lock:
            self._latest_base_frame = safe_frame
            self._latest_version += 1
        return (time.perf_counter() - start) * 1000.0

    def _get_latest(self):
        with self._frame_lock:
            return self._latest_base_frame, self._latest_version

    def _writer_loop(self):
        period = 1.0 / self.fps
        next_tick = time.perf_counter()
        last_written_version = -1

        while not self.stop_event.is_set():
            base_frame, version = self._get_latest()

            if base_frame is None:
                time.sleep(0.005)
                next_tick = time.perf_counter()
                continue

            now = time.perf_counter()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.010))
                continue

            if now - next_tick > period:
                self.deadline_misses += 1
                # Uzun stall sonrasi sonsuz catch-up burst yapma.
                next_tick = now

            out = base_frame.copy()
            draw_server_time(out, self.time_provider.now(), anchor="right", y=32)

            try:
                write_start = time.perf_counter()
                if self.process is None or self.process.stdin is None:
                    break
                self.process.stdin.write(out.tobytes())
                write_ms = (time.perf_counter() - write_start) * 1000.0
                self.write_ms_samples.append(write_ms)
                self.total_written_frames += 1

                if version == last_written_version:
                    self.duplicate_frames += 1
                last_written_version = version

            except (BrokenPipeError, OSError):
                self.broken_pipe = True
                break

            next_tick += period

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None

        if self.process is not None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass

            try:
                self.process.wait(timeout=5.0)
            except Exception:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
            self.process = None

        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None


def metric_stats(values):
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_metric_line(file, name, values, unit="ms"):
    stats = metric_stats(values)
    file.write(
        f"{name}: avg={stats['avg']:.3f} {unit}, "
        f"p50={stats['p50']:.3f} {unit}, "
        f"p95={stats['p95']:.3f} {unit}, "
        f"max={stats['max']:.3f} {unit}, n={stats['count']}\n"
    )


def get_hit_area(frame_w, frame_h, cfg):
    left_ratio = float(cfg.get("hit_left_ratio", DEFAULT_HIT_LEFT_RATIO))
    right_ratio = float(cfg.get("hit_right_ratio", DEFAULT_HIT_RIGHT_RATIO))
    top_ratio = float(cfg.get("hit_top_ratio", DEFAULT_HIT_TOP_RATIO))
    bottom_ratio = float(cfg.get("hit_bottom_ratio", DEFAULT_HIT_BOTTOM_RATIO))

    left_ratio = max(0.0, min(1.0, left_ratio))
    right_ratio = max(0.0, min(1.0, right_ratio))
    top_ratio = max(0.0, min(1.0, top_ratio))
    bottom_ratio = max(0.0, min(1.0, bottom_ratio))

    x1 = int(round(frame_w * left_ratio))
    x2 = int(round(frame_w * right_ratio))
    y1 = int(round(frame_h * top_ratio))
    y2 = int(round(frame_h * bottom_ratio))

    return x1, y1, x2, y2


def point_inside_rect(cx, cy, rect):
    x1, y1, x2, y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2


def detect_frame(model, frame, cfg):
    """Sadece YOLO detect. Tracking/tracker.persist/ID yok."""
    conf_threshold = float(cfg["conf_threshold"])
    iou_threshold = float(cfg["iou_threshold"])
    imgsz = int(cfg["imgsz"])
    device = cfg["device"]

    detect_start = time.perf_counter()

    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    detect_ms = (time.perf_counter() - detect_start) * 1000.0
    detections = []

    if not results:
        return detections, detect_ms

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections, detect_ms

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))

    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "xyxy": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class_id": int(cls_id),
        })

    return detections, detect_ms


def normalize_target_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _get_value(obj, *names, default=None):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def normalize_track_box(item):
    """src.tracker ciktisini ortak bbox/conf/class/track_id formuna cevirir."""
    if item is None:
        return None
    xyxy = _get_value(item, "xyxy", "bbox", default=None)
    if xyxy is None:
        return None
    if hasattr(xyxy, "detach"):
        xyxy = xyxy.detach().cpu().numpy()
    elif hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    elif hasattr(xyxy, "tolist"):
        xyxy = xyxy.tolist()
    arr = np.asarray(xyxy, dtype=float).reshape(-1)
    if arr.size < 4:
        return None
    confidence = _get_value(item, "confidence", "conf", default=0.0)
    class_id = _get_value(item, "class_id", "cls", default=0)
    track_id = _get_value(item, "track_id", "id", "target_id", default=None)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    try:
        class_id = int(class_id)
    except Exception:
        class_id = 0
    return {
        "xyxy": [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])],
        "confidence": confidence,
        "class_id": class_id,
        "track_id": normalize_target_id(track_id),
    }


def evaluate_detection(det, frame_w, frame_h, hit_area, min_axis_ratio):
    x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]

    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(0, min(frame_w - 1, x2))
    y1 = max(0, min(frame_h - 1, y1))
    y2 = max(0, min(frame_h - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bbox_w = max(0, x2 - x1)
    bbox_h = max(0, y2 - y1)
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    
    width_ratio = bbox_w / float(frame_w) if frame_w > 0 else 0.0
    height_ratio = bbox_h / float(frame_h) if frame_h > 0 else 0.0

 
    center_in_hit_area = point_inside_rect(center_x, center_y, hit_area)

# Hedef, tüm ekranın en az %6'sını kaplamalı
    size_ok = (
        width_ratio >= min_axis_ratio
        or height_ratio >= min_axis_ratio
    )

    valid_for_lock = center_in_hit_area and size_ok



    return {
        **det,
        "xyxy_int": [x1, y1, x2, y2],
        "center_x": center_x,
        "center_y": center_y,
        "bbox_width_ratio": width_ratio,
        "bbox_height_ratio": height_ratio,
        "center_in_hit_area": center_in_hit_area,
        "size_ok": size_ok,
        "valid_for_lock": valid_for_lock,
    }


def choose_lock_candidate(evaluated_detections):
    """
    Tracking olmadigi icin ID ile hedef secimi yapmiyoruz.
    Vurus alani + boyut kosulunu saglayan detection'lar arasindan confidence'i
    en yuksek olani o frame'in kilit adayi kabul edilir.
    """
    valid = [d for d in evaluated_detections if d["valid_for_lock"]]
    if not valid:
        return None
    return max(valid, key=lambda d: d["confidence"])


def _bbox_iou(det_a, det_b):
    if det_a is None or det_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = det_a["xyxy_int"]
    bx1, by1, bx2, by2 = det_b["xyxy_int"]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def choose_manual_yolo_continuation(reference_det, raw_detections, frame_w, frame_h,
                                    velocity_px_s=(0.0, 0.0), dt_s=0.0):
    """ByteTrack focus kayboldugunda raw YOLO'da ayni fiziksel hedefin devamini ara.

    Bilerek gevsek tutulur: hizli IHA'yi gereksiz yere LOST yapmamak ana hedeftir.
    IoU zorunlu degildir; tahmini merkeze yakinlik ana kriter, bbox boyutu ve IoU
    yalnizca yardimci skordur. Her frame yeni raw bbox secilir; stale bbox gonderilmez.
    """
    if reference_det is None or not raw_detections:
        return None

    ref_cx = float(reference_det.get("center_x", 0.0))
    ref_cy = float(reference_det.get("center_y", 0.0))
    vx, vy = velocity_px_s
    dt_s = max(0.0, min(float(dt_s), 0.25))
    pred_cx = ref_cx + float(vx) * dt_s
    pred_cy = ref_cy + float(vy) * dt_s

    rx1, ry1, rx2, ry2 = reference_det["xyxy_int"]
    ref_w = max(1.0, float(rx2 - rx1))
    ref_h = max(1.0, float(ry2 - ry1))
    ref_diag = math.hypot(ref_w, ref_h)
    frame_diag = math.hypot(float(frame_w), float(frame_h))

    # Hizli hedef icin genis ama sonsuz olmayan kapı. Amac hedefi kaybetmemek;
    # baska IHA'ya bariz buyuk bir sicramayi yine de engeller.
    max_center_dist = max(140.0, 5.0 * ref_diag, 0.30 * frame_diag)

    best = None
    best_score = None
    for det in raw_detections:
        cx = float(det.get("center_x", 0.0))
        cy = float(det.get("center_y", 0.0))
        center_dist = math.hypot(cx - pred_cx, cy - pred_cy)
        if center_dist > max_center_dist:
            continue

        x1, y1, x2, y2 = det["xyxy_int"]
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        size_penalty = abs(math.log(w / ref_w)) + abs(math.log(h / ref_h))
        iou_bonus = _bbox_iou(reference_det, det)
        confidence_bonus = float(det.get("confidence", 0.0))

        # Dusuk skor daha iyi. Mesafe baskin; size/IoU/confidence sadece yardimci.
        score = (
            center_dist / max(max_center_dist, 1.0)
            + 0.12 * size_penalty
            - 0.08 * iou_bonus
            - 0.03 * confidence_bonus
        )
        if best_score is None or score < best_score:
            best = det
            best_score = score

    return best


def choose_track_matching_raw(raw_det, evaluated_tracks, frame_w, frame_h):
    """Ayni frame'deki raw YOLO fallback kutusuna denk gelen ByteTrack kutusunu bul.

    ByteTrack eski veya yeni ID ile geri geldiginde physical target'a yakin kutu
    otomatik focus edilir. Confidence tek basina ID degistirme nedeni degildir.
    """
    if raw_det is None or not evaluated_tracks:
        return None

    rcx = float(raw_det.get("center_x", 0.0))
    rcy = float(raw_det.get("center_y", 0.0))
    rx1, ry1, rx2, ry2 = raw_det["xyxy_int"]
    raw_diag = math.hypot(max(1.0, rx2 - rx1), max(1.0, ry2 - ry1))
    frame_diag = math.hypot(float(frame_w), float(frame_h))
    max_center_dist = max(60.0, 1.75 * raw_diag, 0.06 * frame_diag)

    best = None
    best_score = None
    for det in evaluated_tracks:
        if det.get("track_id") is None:
            continue
        center_dist = math.hypot(
            float(det.get("center_x", 0.0)) - rcx,
            float(det.get("center_y", 0.0)) - rcy,
        )
        iou = _bbox_iou(raw_det, det)
        if center_dist > max_center_dist and iou <= 0.01:
            continue
        score = center_dist / max(max_center_dist, 1.0) - 0.25 * iou
        if best_score is None or score < best_score:
            best = det
            best_score = score
    return best


def reset_lock_state():
    return None, None, False, 0.0, 0.0, None


# ---------------- LOCK ENGINE: mevcut main loop mantiginin mode-aware hali ----------------

lock_target_pub = None
hedef_piksel_pub = None
lock_success_pub = None
lock_ros_active = False
lock_target_topic = DEFAULT_TARGET_DATA_TOPIC
hedef_piksel_topic = DEFAULT_HEDEF_PIKSEL_TOPIC
lock_success_topic = DEFAULT_LOCK_TOPIC
lock_duration_s_global = DEFAULT_LOCK_DURATION_S


def init_lock_ros_after_qr(cfg):
    """QR init_node'dan SONRA ayni ROS node icinde LOCK publisher'larini olusturur."""
    global lock_target_pub, hedef_piksel_pub, lock_success_pub, lock_ros_active
    global lock_target_topic, hedef_piksel_topic, lock_success_topic, lock_duration_s_global

    lock_target_topic = str(cfg.get("target_data_topic", DEFAULT_TARGET_DATA_TOPIC))
    hedef_piksel_topic = DEFAULT_HEDEF_PIKSEL_TOPIC
    lock_success_topic = str(cfg.get("lock_topic", DEFAULT_LOCK_TOPIC))
    lock_duration_s_global = float(cfg.get("lock_duration_s", DEFAULT_LOCK_DURATION_S))

    if not USE_ROS or rospy is None or String is None:
        lock_ros_active = False
        lock_target_pub = None
        hedef_piksel_pub = None
        lock_success_pub = None
        print("[INFO] LOCK ROS kullanilamiyor; terminal modu.")
        return

    try:
        lock_target_topic = rospy.get_param("~target_data_topic", lock_target_topic)
        lock_success_topic = rospy.get_param("~lock_topic", lock_success_topic)
        lock_duration_s_global = float(rospy.get_param("~lock_duration_s", lock_duration_s_global))
        lock_target_pub = rospy.Publisher(lock_target_topic, String, queue_size=10)
        hedef_piksel_pub = rospy.Publisher(hedef_piksel_topic, String, queue_size=10)
        lock_success_pub = rospy.Publisher(lock_success_topic, String, queue_size=10)
        lock_ros_active = True
        print(f"[INFO] LOCK ROS Detection topic: {lock_target_topic}")
        print(f"[INFO] LOCK ROS Hedef piksel topic: {hedef_piksel_topic}")
        print(f"[INFO] LOCK ROS Kilit topic: {lock_success_topic}")
        print(f"[INFO] Kilit suresi: {lock_duration_s_global:.2f} s")
    except Exception as exc:
        lock_ros_active = False
        lock_target_pub = None
        hedef_piksel_pub = None
        lock_success_pub = None
        print(f"[WARN] LOCK publisher baslatilamadi: {exc}")


class LockEngine:
    """
    K modu = kullanicinin yeni EVERY-FRAME YOLO + ByteTrack + ROS LOCK kodu.

    Degismeyen combined mimari:
      - Herelink RTSP PersistentVideoBackbone'da TEK client.
      - Bu engine yalniz ortak local decoder'dan gelen lock_frame_queue'yu tuketir.
      - Q/QR, raw, GUI, evaluation ve unified log mimarisi aynen korunur.
    """
    def __init__(
        self,
        cfg,
        tracker,
        evaluation_recorder,
        server_time_provider,
        csv_path,
        report_path,
        backbone,
    ):
        self.cfg = cfg
        self.tracker = tracker
        self.evaluation_recorder = evaluation_recorder
        self.server_time_provider = server_time_provider
        self.csv_path = Path(csv_path)
        self.report_path = Path(report_path)
        self.backbone = backbone

        self.frame_w = int(W)
        self.frame_h = int(H)
        self.input_fps = max(1.0, float(cfg.get("capture_fps", DEFAULT_CAPTURE_FPS)))
        self.min_axis_ratio = max(
            0.0,
            min(1.0, float(cfg.get("min_target_axis_ratio", DEFAULT_MIN_TARGET_AXIS_RATIO))),
        )
        self.hit_area = get_hit_area(self.frame_w, self.frame_h, cfg)
        self.lock_duration_s = float(lock_duration_s_global)

        self.thread = None
        self.local_stop = threading.Event()
        self.manual_timer_reset_request = threading.Event()
        self.last_mode_generation = -1

        self.frame_id = 0
        self.total_frames = 0
        self.detection_frames = 0
        self.valid_candidate_frames = 0
        self.lock_success_count = 0
        self.manual_publish_frames = 0
        self.manual_focus_events = 0
        self.tracking_status_counts = {}

        self.latency_samples = {
            "read_wait_ms": [],
            "capture_pipe_read_ms": [],
            "capture_interval_ms": [],
            "ingest_queue_ms": [],
            "queue_wait_ms": [],
            "yolo_track_ms": [],
            "processing_ms": [],
            "frame_age_ms": [],
            "evaluation_build_ms": [],
            "evaluation_update_ms": [],
            "debug_draw_ms": [],
            "frame_total_ms": [],
            "display_ms": [],
            "q_value": [],
        }

        self._manual_lock = threading.Lock()
        self._reset_tracking_session(reset_tracker=True)
        self._open_csv()

    def _new_state_manager(self):
        return TargetStateManager(
            temporary_lost_frames=int(self.cfg["temporary_lost_frames"]),
            lost_frames=int(self.cfg["lost_frames"]),
            reset_frames=int(self.cfg["reset_frames"]),
        )

    def _try_reset_tracker(self):
        """ByteTrack state'ini yeni K oturumu icin sifirlamaya calis; modeli yeniden yukleme."""
        candidates = [
            self.tracker,
            getattr(self.tracker, "tracker", None),
            getattr(self.tracker, "byte_tracker", None),
        ]
        for obj in candidates:
            if obj is None:
                continue
            reset_fn = getattr(obj, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                    return True
                except Exception:
                    pass
        return False

    def _reset_tracking_session(self, reset_tracker=False):
        # Yeni K acilisi, standalone kodun yeni calistirilmasi gibi temiz tracking state ile baslar.
        self.state_manager = self._new_state_manager()
        if reset_tracker:
            self._try_reset_tracker()

        self.prev_time = None
        self.prev_error_x = None
        self.prev_error_y = None
        self.manual_prev_error_x = None
        self.manual_prev_error_y = None

        with self._manual_lock:
            self.manual_target_publish = False
            self.selected_target_id = None
            self.focused_target_id = None
            self._selectable_tracks_snapshot = []
            self._manual_focus_center_snapshot = None
            self._manual_reference_det = None
            self._manual_reference_time = None
            self._manual_velocity_px_s = (0.0, 0.0)
            self._manual_fallback_active = False
            self._manual_blind_since_monotonic = None

        self.manual_timer_reset_request.clear()
        self._reset_lock_state()

    def _clear_manual_continuity_locked(self):
        self._manual_focus_center_snapshot = None
        self._manual_reference_det = None
        self._manual_reference_time = None
        self._manual_velocity_px_s = (0.0, 0.0)
        self._manual_fallback_active = False
        self._manual_blind_since_monotonic = None

    def _update_manual_reference_locked(self, det, now_mono=None, reset_velocity=False):
        if det is None:
            return

        # Hedef tekrar goruldu: kisa blind-hold durumu biter.
        self._manual_blind_since_monotonic = None

        if now_mono is None:
            now_mono = time.monotonic()

        cx = float(det.get("center_x", 0.0))
        cy = float(det.get("center_y", 0.0))
        if (
            not reset_velocity
            and self._manual_reference_det is not None
            and self._manual_reference_time is not None
        ):
            dt_ref = max(float(now_mono) - float(self._manual_reference_time), 1e-6)
            prev_cx = float(self._manual_reference_det.get("center_x", cx))
            prev_cy = float(self._manual_reference_det.get("center_y", cy))
            inst_vx = (cx - prev_cx) / dt_ref
            inst_vy = (cy - prev_cy) / dt_ref
            old_vx, old_vy = self._manual_velocity_px_s
            self._manual_velocity_px_s = (
                0.55 * float(old_vx) + 0.45 * inst_vx,
                0.55 * float(old_vy) + 0.45 * inst_vy,
            )
        else:
            self._manual_velocity_px_s = (0.0, 0.0)

        self._manual_reference_det = dict(det)
        self._manual_reference_time = float(now_mono)
        self._manual_focus_center_snapshot = (int(round(cx)), int(round(cy)))

    def _open_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame", "camera_frame_id", "source_sequence", "timestamp",
            "server_time_source", "server_time",
            "state_target_id", "tracking_status", "target_valid", "track_count",
            "selected_target_id", "focused_target_id", "manual_publish_enabled",
            "q_value", "queue_wait_ms", "frame_age_ms",
            "manual_confidence", "manual_normalized_error_x", "manual_normalized_error_y",
            "manual_target_velocity_x", "manual_target_velocity_y", "manual_bbox_area_ratio",
            "manual_center_x", "manual_center_y",
            "manual_bbox_x1", "manual_bbox_y1", "manual_bbox_x2", "manual_bbox_y2",
            "manual_bbox_width_ratio", "manual_bbox_height_ratio",
            "manual_center_in_hit_area", "manual_size_ok",
            "lock_target_id", "lock_candidate_confidence", "lock_candidate_valid",
            "lock_elapsed_s", "tolerance_used_ms", "lock_success",
            "yolo_track_ms", "processing_ms", "fps",
            "read_wait_ms", "capture_pipe_read_ms", "capture_interval_ms",
            "ingest_queue_ms", "evaluation_build_ms", "evaluation_update_ms",
            "debug_draw_ms", "frame_total_ms", "evaluation_written_frames",
            "evaluation_duplicate_frames", "backbone_restart_count",
            "vision_decoder_restart_count",
        ])

    def _reset_lock_state(self):
        (
            self.lock_start_monotonic,
            self.lock_target_id,
            self.lock_success_sent,
            self.lock_elapsed_s,
            self.lock_missing_total_s,
            self.lock_missing_since_monotonic,
        ) = reset_lock_state()

    def start(self):
        self.local_stop.clear()
        self.thread = threading.Thread(target=self._run, name="lock_engine_bytetrack", daemon=True)
        self.thread.start()

    def handle_key(self, key):
        """K modunda B/N/SPACE manuel hedef secimini ve '-' ile timer resetini uygular."""
        if get_active_mode() != MODE_LOCK:
            return False

        key = int(key) & 0xFF
        with self._manual_lock:
            selectable_tracks = [dict(d) for d in self._selectable_tracks_snapshot]
            selected_target_id = self.selected_target_id
            manual_target_publish = self.manual_target_publish

            if key in (ord('b'), ord('B'), ord('n'), ord('N')):
                go_left = key in (ord('b'), ord('B'))

                # SPACE OFF: mavi secili hedeften B/N ile gezin.
                # SPACE ON + YESIL: aktif focused hedeften direkt diger track'e gec.
                # SPACE ON + MOR: son fallback merkezini referans alip gorunen ByteTrack hedefe gec.
                reference_x = None
                current_id = self.focused_target_id if manual_target_publish else selected_target_id
                current_det = next(
                    (d for d in selectable_tracks
                     if normalize_target_id(d.get("track_id")) == current_id),
                    None,
                )
                if current_det is not None:
                    reference_x = float(current_det["center_x"])
                elif manual_target_publish and self._manual_focus_center_snapshot is not None:
                    reference_x = float(self._manual_focus_center_snapshot[0])

                if reference_x is not None:
                    if go_left:
                        side_tracks = [d for d in selectable_tracks if float(d["center_x"]) < reference_x]
                        new_det = max(side_tracks, key=lambda d: float(d["center_x"])) if side_tracks else None
                    else:
                        side_tracks = [d for d in selectable_tracks if float(d["center_x"]) > reference_x]
                        new_det = min(side_tracks, key=lambda d: float(d["center_x"])) if side_tracks else None

                    if new_det is not None:
                        new_id = normalize_target_id(new_det["track_id"])
                        self.selected_target_id = new_id
                        if manual_target_publish:
                            self.focused_target_id = new_id
                            self._manual_fallback_active = False
                            self._update_manual_reference_locked(new_det, reset_velocity=True)
                        self.manual_prev_error_x = None
                        self.manual_prev_error_y = None
                        direction = "SOL" if go_left else "SAG"
                        mode_text = "FOCUS" if manual_target_publish else "SECIM"
                        print(f"[MANUAL] {direction} hedef | {mode_text} ID={new_id}")
                        log_system_event(
                            "manual_focus_switch" if manual_target_publish else ("manual_select_left" if go_left else "manual_select_right"),
                            f"id={new_id},direction={direction}",
                            frame_id=latest_frame_id,
                        )
                        return True

            elif key == 32:  # SPACE
                if not manual_target_publish:
                    # SPACE acilirken eski/stale ID kabul etme; secili hedef su an gorunur olmali.
                    selected_det = next(
                        (d for d in selectable_tracks
                         if normalize_target_id(d.get("track_id")) == self.selected_target_id),
                        None,
                    )
                    if selected_det is None and selectable_tracks:
                        selected_det = max(
                            selectable_tracks,
                            key=lambda d: float(d.get("confidence", 0.0)),
                        )
                        self.selected_target_id = normalize_target_id(selected_det["track_id"])

                    if selected_det is not None:
                        self.focused_target_id = normalize_target_id(selected_det["track_id"])
                        self.selected_target_id = self.focused_target_id
                        self.manual_target_publish = True
                        self.manual_prev_error_x = None
                        self.manual_prev_error_y = None
                        self._manual_fallback_active = False
                        self._update_manual_reference_locked(selected_det, reset_velocity=True)
                        self.manual_focus_events += 1
                        print(
                            f"[MANUAL] FOCUS ACIK | ID={self.focused_target_id} | "
                            f"{lock_target_topic} yayin: ACIK"
                        )
                        log_system_event("manual_focus_on", f"id={self.focused_target_id}", frame_id=latest_frame_id)
                        return True
                else:
                    old_id = self.focused_target_id
                    self.manual_target_publish = False
                    self.focused_target_id = None
                    self.selected_target_id = None
                    self.manual_prev_error_x = None
                    self.manual_prev_error_y = None
                    self._clear_manual_continuity_locked()

                    # Eski focus tamamen unutulur. O anda gorunen hedeflerden yeni mavi aday secilir.
                    if selectable_tracks:
                        new_det = max(
                            selectable_tracks,
                            key=lambda d: float(d.get("confidence", 0.0)),
                        )
                        self.selected_target_id = normalize_target_id(new_det["track_id"])

                    self.manual_focus_events += 1
                    print(
                        f"[MANUAL] FOCUS KAPALI | eski ID={old_id} unutuldu | "
                        f"yeni secili ID={self.selected_target_id} | {lock_target_topic} yayin: KAPALI"
                    )
                    log_system_event(
                        "manual_focus_off",
                        f"old_id={old_id},new_selected_id={self.selected_target_id}",
                        frame_id=latest_frame_id,
                    )
                    return True

            elif key == ord('-'):
                # Yanlis/negatif tespitte sadece kilit timer'ini sifirla.
                # Focus, ByteTrack secimi ve /target_data yayin durumuna dokunma.
                self.manual_timer_reset_request.set()
                self._reset_lock_state()
                print("[LOCK] TIMER MANUEL SIFIRLANDI (-)")
                log_system_event("manual_lock_timer_reset", "key=-", frame_id=latest_frame_id)
                return True
        return False

    def _run(self):
        global latest_lock_display_frame
        print("[INFO] LOCK ByteTrack engine thread basladi; K tusunu bekliyor.")

        while not stop_event.is_set() and not self.local_stop.is_set() and not ros_is_shutdown():
            if get_active_mode() != MODE_LOCK:
                self._reset_lock_state()
                self.last_mode_generation = get_mode_generation()
                time.sleep(0.01)
                continue

            current_gen = get_mode_generation()
            if current_gen != self.last_mode_generation:
                self.last_mode_generation = current_gen
                _drain_queue(lock_frame_queue)
                self._reset_tracking_session(reset_tracker=True)
                print("[INFO] Yeni K oturumu: ByteTrack/TargetState/lock/manual state sifirlandi.")
                log_system_event("lock_session_reset", f"generation={current_gen}", frame_id=latest_frame_id)

            read_wait_start = time.perf_counter()
            try:
                item = lock_frame_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            read_wait_ms = (time.perf_counter() - read_wait_start) * 1000.0

            try:
                if get_active_mode() != MODE_LOCK:
                    continue
                camera_fid, clean_frame, capture_done_mono, capture_pipe_read_ms, capture_interval_ms = item
                self._process_one(
                    camera_fid,
                    clean_frame,
                    capture_done_mono,
                    capture_pipe_read_ms,
                    capture_interval_ms,
                    read_wait_ms,
                )
            finally:
                lock_frame_queue.task_done()

        print("[INFO] LOCK ByteTrack engine thread kapandi.")

    def _process_one(
        self,
        camera_fid,
        clean_frame,
        capture_done_mono,
        capture_pipe_read_ms,
        capture_interval_ms,
        read_wait_ms,
    ):
        global latest_lock_display_frame

        frame_start = time.perf_counter()
        now_wall = time.time()
        timestamp_iso = datetime.now().isoformat(timespec="milliseconds")
        queue_wait_ms = max(0.0, (time.perf_counter() - capture_done_mono) * 1000.0)
        ingest_queue_ms = queue_wait_ms

        if self.prev_time is None:
            dt = 1.0 / self.input_fps
        else:
            dt = max(now_wall - self.prev_time, 1e-6)

        # ============================================================
        # TEK YOLO INFERENCE, IKI BAGIMSIZ YOL:
        #   1) raw_detections -> LOCK / timer / %6 / vurus alani
        #   2) track_boxes    -> sadece MPC hedef secimi + /target_data
        # ByteTrack LOCK kararini veya timer'i ETKILEMEZ.
        # ============================================================
        process_generation = get_mode_generation()
        detections, track_boxes, yolo_track_ms = self.tracker.track_with_detections(clean_frame)

        # Tracking state sadece MPC/ID tarafinda kullanilir; LOCK/timer bundan bagimsizdir.
        target, tracking_status, target_valid = self.state_manager.update(track_boxes)

        if get_active_mode() != MODE_LOCK or get_mode_generation() != process_generation:
            return

        current_target_id = normalize_target_id(
            getattr(self.state_manager, "current_target_id", None)
        )

        # ONLY-FRAME kodundaki gibi ham YOLO detection'lari.
        evaluated = []
        if detections is not None:
            for det in detections:
                evaluated.append(
                    evaluate_detection(
                        det,
                        self.frame_w,
                        self.frame_h,
                        self.hit_area,
                        self.min_axis_ratio,
                    )
                )

        # ByteTrack ciktilari SADECE manuel MPC hedef secimi / target_data icin.
        evaluated_tracks = []
        if track_boxes is not None:
            for tb in track_boxes:
                det = normalize_track_box(tb)
                if det is None:
                    continue
                evaluated_tracks.append(
                    evaluate_detection(
                        det,
                        self.frame_w,
                        self.frame_h,
                        self.hit_area,
                        self.min_axis_ratio,
                    )
                )

        if evaluated:
            self.detection_frames += 1

        selectable_tracks = [
            det for det in evaluated_tracks if det.get("track_id") is not None
        ]

        # Manuel secim/focus state LOCK candidate'inden tamamen bagimsiz.
        with self._manual_lock:
            self._selectable_tracks_snapshot = [dict(d) for d in selectable_tracks]
            selected_target_visible = any(
                normalize_target_id(det.get("track_id")) == self.selected_target_id
                for det in selectable_tracks
            )
            if not self.manual_target_publish and self.focused_target_id is None:
                if self.selected_target_id is None or not selected_target_visible:
                    if selectable_tracks:
                        selected_det = max(
                            selectable_tracks,
                            key=lambda det: float(det.get("confidence", 0.0)),
                        )
                        self.selected_target_id = normalize_target_id(selected_det["track_id"])
                        self.manual_prev_error_x = None
                        self.manual_prev_error_y = None
                    else:
                        self.selected_target_id = None
            selected_target_id = self.selected_target_id
            focused_target_id = self.focused_target_id
            manual_target_publish = self.manual_target_publish

        # LOCK candidate = track'siz ONLY-FRAME kodundaki gibi dogrudan YOLO detection.
        candidate = choose_lock_candidate(evaluated)
        if candidate is not None:
            self.valid_candidate_frames += 1

        # ============================================================
        # MANUAL FOCUS SUREKLILIGI
        # - ByteTrack focused ID varsa: YESIL / normal /target_data.
        # - ID kaybolursa: son fiziksel hedefe en uygun raw YOLO detection MOR fallback.
        # - Ayni fiziksel hedef ByteTrack'te eski/yeni ID ile donerse otomatik rebind.
        # - Sabit fallback timeout YOK; raw YOLO hedef devam ettigi surece publish devam.
        # - Hem focused ByteTrack hem de ayni raw YOLO hedef yoksa focus/publish kapanir.
        # ============================================================
        manual_candidate = None
        manual_fallback_candidate = None
        manual_is_fallback = False
        manual_now_mono = time.monotonic()

        with self._manual_lock:
            focused_target_id = self.focused_target_id
            manual_target_publish = self.manual_target_publish
            reference_det = dict(self._manual_reference_det) if self._manual_reference_det is not None else None
            reference_time = self._manual_reference_time
            reference_velocity = tuple(self._manual_velocity_px_s)
            fallback_was_active = bool(self._manual_fallback_active)

        if manual_target_publish and focused_target_id is not None:
            exact_track = next(
                (det for det in evaluated_tracks
                 if normalize_target_id(det.get("track_id")) == focused_target_id),
                None,
            )

            # Normal ACTIVE durumda mevcut focused ID devam ediyorsa direkt kullan.
            if exact_track is not None and not fallback_was_active:
                manual_candidate = exact_track
                with self._manual_lock:
                    if self.manual_target_publish and self.focused_target_id == focused_target_id:
                        self._manual_fallback_active = False
                        self._update_manual_reference_locked(exact_track, manual_now_mono)
            else:
                # ByteTrack kaybi veya MOR fallback devaminda raw YOLO fiziksel surekliligini ara.
                dt_reference = 0.0
                if reference_time is not None:
                    dt_reference = max(0.0, manual_now_mono - float(reference_time))
                raw_continuation = choose_manual_yolo_continuation(
                    reference_det,
                    evaluated,
                    self.frame_w,
                    self.frame_h,
                    velocity_px_s=reference_velocity,
                    dt_s=dt_reference,
                )

                if raw_continuation is not None:
                    # ByteTrack ayni fiziksel hedefi eski veya yeni ID ile geri yakaladiysa
                    # raw kutuya denk gelen track'e otomatik rebind et ve YESIL'e don.
                    rebound_track = choose_track_matching_raw(
                        raw_continuation, evaluated_tracks, self.frame_w, self.frame_h
                    )
                    if rebound_track is not None:
                        rebound_id = normalize_target_id(rebound_track.get("track_id"))
                        manual_candidate = rebound_track
                        with self._manual_lock:
                            if self.manual_target_publish:
                                old_focus_id = self.focused_target_id
                                self.focused_target_id = rebound_id
                                self.selected_target_id = rebound_id
                                self._manual_fallback_active = False
                                self._update_manual_reference_locked(rebound_track, manual_now_mono)
                                focused_target_id = rebound_id
                                if old_focus_id != rebound_id:
                                    print(f"[MANUAL] HEDEF YENIDEN YAKALANDI | ID {old_focus_id} -> {rebound_id}")
                                    log_system_event(
                                        "manual_focus_rebind",
                                        f"old_id={old_focus_id},new_id={rebound_id}",
                                        frame_id=camera_fid,
                                    )
                    else:
                        # ByteTrack yok ama raw YOLO fiziksel hedefi goruyor: MOR fallback.
                        manual_candidate = raw_continuation
                        manual_fallback_candidate = raw_continuation
                        manual_is_fallback = True
                        with self._manual_lock:
                            if self.manual_target_publish:
                                self._manual_fallback_active = True
                                self._update_manual_reference_locked(raw_continuation, manual_now_mono)
                                focused_target_id = self.focused_target_id
                elif exact_track is not None:
                    # Nadir durumda raw callback bos olsa bile focused ByteTrack gercekten mevcutsa
                    # track'i kaybetmis sayma.
                    manual_candidate = exact_track
                    with self._manual_lock:
                        if self.manual_target_publish:
                            self._manual_fallback_active = False
                            self._update_manual_reference_locked(exact_track, manual_now_mono)
                else:
                    # Ne focused ByteTrack ne de ayni fiziksel hedefe uyan raw YOLO var.
                    # Hemen LOST yapma: timer toleransi ile ayni 200 ms boyunca focus
                    # hafizada tutulur. Bu surede manual_candidate=None oldugu icin
                    # stale bbox /target_data olarak GONDERILMEZ.
                    with self._manual_lock:
                        if self.manual_target_publish:
                            if self._manual_blind_since_monotonic is None:
                                self._manual_blind_since_monotonic = manual_now_mono

                            blind_elapsed_s = (
                                manual_now_mono - self._manual_blind_since_monotonic
                            )

                            if blind_elapsed_s <= DEFAULT_LOCK_LOSS_TOLERANCE_S:
                                # Kisa goruntu boslugu: SPACE/focus hafizada kalsin.
                                # Paket yayinlanmaz; hedef geri gelirse YESIL veya MOR devam eder.
                                focused_target_id = self.focused_target_id
                                manual_target_publish = True
                            else:
                                # 200 ms boyunca ne ByteTrack ne raw YOLO geri geldi: gercek LOST.
                                lost_id = self.focused_target_id
                                self.manual_target_publish = False
                                self.focused_target_id = None
                                self.selected_target_id = None
                                self.manual_prev_error_x = None
                                self.manual_prev_error_y = None
                                self._clear_manual_continuity_locked()
                                if selectable_tracks:
                                    new_det = max(
                                        selectable_tracks,
                                        key=lambda d: float(d.get("confidence", 0.0)),
                                    )
                                    self.selected_target_id = normalize_target_id(new_det["track_id"])
                                print(f"[MANUAL] FOCUS LOST | ID={lost_id} | {lock_target_topic} yayin: KAPALI")
                                log_system_event(
                                    "manual_focus_lost", f"id={lost_id}", frame_id=camera_fid
                                )
                        focused_target_id = self.focused_target_id
                        manual_target_publish = self.manual_target_publish

        # Son focus/publish state'ini bu frame'in cizim/yayin kismi icin tazele.
        with self._manual_lock:
            selected_target_id = self.selected_target_id
            focused_target_id = self.focused_target_id
            manual_target_publish = self.manual_target_publish

        # StateManager target velocity/error state.
        normalized_error_x = 0.0
        normalized_error_y = 0.0
        target_velocity_x = 0.0
        target_velocity_y = 0.0
        bbox_area_ratio = 0.0
        if target_valid and target is not None:
            outputs = calculate_tracking_outputs(
                target=target,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                prev_error_x=self.prev_error_x,
                prev_error_y=self.prev_error_y,
                dt=dt,
            )
            normalized_error_x = float(outputs["normalized_error_x"])
            normalized_error_y = float(outputs["normalized_error_y"])
            target_velocity_x = float(outputs["target_velocity_x"])
            target_velocity_y = float(outputs["target_velocity_y"])
            bbox_area_ratio = float(outputs["bbox_area_ratio"])
            self.prev_error_x = normalized_error_x
            self.prev_error_y = normalized_error_y
        else:
            self.prev_error_x = None
            self.prev_error_y = None

        # Manuel /target_data verileri sadece focused_target_id'den.
        manual_confidence = 0.0
        manual_normalized_error_x = 0.0
        manual_normalized_error_y = 0.0
        manual_target_velocity_x = 0.0
        manual_target_velocity_y = 0.0
        manual_bbox_area_ratio = 0.0
        manual_center_x = 0
        manual_center_y = 0
        manual_bbox_payload = [0, 0, 0, 0]
        manual_width_ratio = 0.0
        manual_height_ratio = 0.0
        manual_center_in_hit_area = False
        manual_size_ok = False

        if manual_candidate is not None:
            manual_outputs = calculate_tracking_outputs(
                target=manual_candidate,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                prev_error_x=self.manual_prev_error_x,
                prev_error_y=self.manual_prev_error_y,
                dt=dt,
            )
            manual_confidence = float(manual_candidate["confidence"])
            manual_normalized_error_x = float(manual_outputs["normalized_error_x"])
            manual_normalized_error_y = float(manual_outputs["normalized_error_y"])
            manual_target_velocity_x = float(manual_outputs["target_velocity_x"])
            manual_target_velocity_y = float(manual_outputs["target_velocity_y"])
            manual_bbox_area_ratio = float(manual_outputs["bbox_area_ratio"])
            manual_center_x = int(manual_candidate["center_x"])
            manual_center_y = int(manual_candidate["center_y"])
            manual_bbox_payload = [int(v) for v in manual_candidate["xyxy_int"]]
            manual_width_ratio = float(manual_candidate["bbox_width_ratio"])
            manual_height_ratio = float(manual_candidate["bbox_height_ratio"])
            manual_center_in_hit_area = bool(manual_candidate["center_in_hit_area"])
            manual_size_ok = bool(manual_candidate["size_ok"])
            self.manual_prev_error_x = manual_normalized_error_x
            self.manual_prev_error_y = manual_normalized_error_y
        else:
            self.manual_prev_error_x = None
            self.manual_prev_error_y = None

        # Ayni TEK lock candidate: timer + evaluation videosu + /hedef_piksel.
        # Candidate yoksa /hedef_piksel paketi GONDERILMEZ. SPACE'ten tamamen bagimsizdir.
        if candidate is not None:
            cx = int(candidate["center_x"])
            cy = int(candidate["center_y"])
            x1, y1, x2, y2 = [int(v) for v in candidate["xyxy_int"]]
            hedef_piksel_payload = {
                "hedef_merkez_X": cx,
                "hedef_merkez_Y": cy,
                "hedef_genislik": max(0, x2 - x1),
                "hedef_yukseklik": max(0, y2 - y1),
            }
            publish_json(
                lock_ros_active,
                hedef_piksel_pub,
                hedef_piksel_topic,
                hedef_piksel_payload,
                terminal_log=False,
            )

        # Musabaka evaluation videosu: sadece timer'in kullandigi TEK candidate kirmizi.
        eval_build_start = time.perf_counter()
        evaluation_base = build_evaluation_base_frame(clean_frame, candidate)
        evaluation_build_ms = (time.perf_counter() - eval_build_start) * 1000.0
        evaluation_update_ms = 0.0
        if self.evaluation_recorder is not None:
            evaluation_update_ms = self.evaluation_recorder.update(evaluation_base)

        # Debug/canli K ekrani.
        frame = clean_frame.copy()
        debug_draw_start = time.perf_counter()
        hx1, hy1, hx2, hy2 = self.hit_area
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), HIT_AREA_COLOR_BGR, HIT_AREA_THICKNESS)
        cv2.putText(
            frame, "VURUS ALANI", (hx1 + 8, max(25, hy1 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, HIT_AREA_COLOR_BGR, 2, cv2.LINE_AA,
        )

        for det in evaluated:
            x1, y1, x2, y2 = det["xyxy_int"]
            is_lock_candidate = candidate is not None and det is candidate
            box_color = VALID_BOX_COLOR_BGR if is_lock_candidate else INVALID_BOX_COLOR_BGR
            center_color = VALID_CENTER_COLOR_BGR if is_lock_candidate else INVALID_CENTER_COLOR_BGR
            box_thickness = SELECTED_BOX_THICKNESS if is_lock_candidate else DETECTION_BOX_THICKNESS
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
            cv2.circle(frame, (det["center_x"], det["center_y"]), 4, center_color, -1)
            cv2.putText(
                frame, f"{det['confidence']:.2f}", (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1, cv2.LINE_AA,
            )

        # ============================================================
        # LOCK / TIMER = SADECE HAM YOLO DETECTION.
        # ByteTrack ID, StateManager ve SPACE timer'i ASLA etkilemez.
        # 200 ms tolerans da track'siz ONLY-FRAME kodundaki gibi:
        # sadece YOLO'nun hic detection uretmedigi kisa bosluklarda kullanilir.
        # ============================================================
        if self.manual_timer_reset_request.is_set():
            self.manual_timer_reset_request.clear()
            self._reset_lock_state()

        now_mono = time.monotonic()
        self.lock_target_id = None  # sadece eski CSV/payload alani; timer artik ID tutmaz.

        if candidate is not None:
            if self.lock_start_monotonic is None:
                self.lock_start_monotonic = now_mono
                self.lock_success_sent = False
                self.lock_elapsed_s = 0.0
                self.lock_missing_total_s = 0.0
                self.lock_missing_since_monotonic = None
            else:
                if self.lock_missing_since_monotonic is not None:
                    self.lock_missing_total_s += now_mono - self.lock_missing_since_monotonic
                    self.lock_missing_since_monotonic = None
                    if self.lock_missing_total_s > DEFAULT_LOCK_LOSS_TOLERANCE_S:
                        self.lock_start_monotonic = now_mono
                        self.lock_success_sent = False
                        self.lock_elapsed_s = 0.0
                        self.lock_missing_total_s = 0.0

            self.lock_elapsed_s = now_mono - self.lock_start_monotonic
            x1, y1, x2, y2 = candidate["xyxy_int"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), SELECTED_BOX_COLOR_BGR, SELECTED_BOX_THICKNESS)

            if self.lock_elapsed_s >= self.lock_duration_s:
                self.lock_elapsed_s = self.lock_duration_s
                if not self.lock_success_sent:
                    publish_lock_success(
                        lock_ros_active,
                        lock_success_pub,
                        lock_success_topic,
                        self.server_time_provider.now(),
                    )
                    self.lock_success_sent = True
                    self.lock_success_count += 1
        else:
            if len(evaluated) == 0:
                # YOLO hic detection vermedi: toplam 200 ms'ye kadar timer korunur.
                if self.lock_start_monotonic is not None:
                    if self.lock_missing_since_monotonic is None:
                        self.lock_missing_since_monotonic = now_mono
                    current_missing_total_s = (
                        self.lock_missing_total_s + (now_mono - self.lock_missing_since_monotonic)
                    )
                    if current_missing_total_s > DEFAULT_LOCK_LOSS_TOLERANCE_S:
                        self._reset_lock_state()
                    else:
                        self.lock_elapsed_s = now_mono - self.lock_start_monotonic
            else:
                # Detection var ama vurus alani/%6 sarti yok: tolerans uygulanmaz.
                self._reset_lock_state()

        # Manuel ByteTrack dis cerceveleri: siyah=diger, camgobegi=secili, yesil=aktif focus.
        for det in evaluated_tracks:
            det_id = normalize_target_id(det.get("track_id"))
            if manual_target_publish and not manual_is_fallback and focused_target_id is not None and det_id == focused_target_id:
                manual_box_color = MANUAL_FOCUSED_COLOR_BGR
                manual_box_thickness = SELECTED_BOX_THICKNESS
            elif not manual_target_publish and selected_target_id is not None and det_id == selected_target_id:
                manual_box_color = MANUAL_SELECTED_COLOR_BGR
                manual_box_thickness = SELECTED_BOX_THICKNESS
            else:
                manual_box_color = MANUAL_OTHER_COLOR_BGR
                manual_box_thickness = DETECTION_BOX_THICKNESS
            x1, y1, x2, y2 = det["xyxy_int"]
            margin = 4
            mx1 = max(0, x1 - margin)
            my1 = max(0, y1 - margin)
            mx2 = min(self.frame_w - 1, x2 + margin)
            my2 = min(self.frame_h - 1, y2 + margin)
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), manual_box_color, manual_box_thickness)

        # ByteTrack focused hedefi yoksa ama raw YOLO fiziksel hedefi devam ettiriyorsa
        # ayri katmanda MOR dis cerceve ciz. Raw detection kutularina dokunmaz.
        if manual_target_publish and manual_is_fallback and manual_fallback_candidate is not None:
            x1, y1, x2, y2 = manual_fallback_candidate["xyxy_int"]
            margin = 7
            mx1 = max(0, x1 - margin)
            my1 = max(0, y1 - margin)
            mx2 = min(self.frame_w - 1, x2 + margin)
            my2 = min(self.frame_h - 1, y2 + margin)
            cv2.rectangle(
                frame, (mx1, my1), (mx2, my2),
                MANUAL_FALLBACK_COLOR_BGR, SELECTED_BOX_THICKNESS,
            )

        q_value = int(lock_frame_queue.qsize())
        frame_age_ms = max(0.0, (time.perf_counter() - capture_done_mono) * 1000.0)
        tolerance_used_s = self.lock_missing_total_s
        if self.lock_missing_since_monotonic is not None:
            tolerance_used_s += max(0.0, now_mono - self.lock_missing_since_monotonic)
        tolerance_used_ms = min(
            int(round(tolerance_used_s * 1000.0)),
            int(round(DEFAULT_LOCK_LOSS_TOLERANCE_S * 1000.0)),
        )
        tolerance_limit_ms = int(round(DEFAULT_LOCK_LOSS_TOLERANCE_S * 1000.0))

        processing_ms = (time.perf_counter() - frame_start) * 1000.0
        fps = 1000.0 / processing_ms if processing_ms > 0 else 0.0

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, FPS_TEXT_COLOR_BGR, 2, cv2.LINE_AA,
        )
        if parse_bool(self.cfg.get("show_latency_debug", True)):
            cv2.putText(frame, f"YOLO+TRACK: {yolo_track_ms:.1f} ms", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, FPS_TEXT_COLOR_BGR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"QUEUE WAIT: {queue_wait_ms:.1f} ms", (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, FPS_TEXT_COLOR_BGR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"FRAME AGE: {frame_age_ms:.1f} ms", (15, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.52, FPS_TEXT_COLOR_BGR, 1, cv2.LINE_AA)

        timer_text = (
            f"Kilitlenme: {self.lock_elapsed_s:.2f}/{self.lock_duration_s:.2f}s "
            f"| Tol: {tolerance_used_ms}/{tolerance_limit_ms}ms"
        )
        q_text = f"Q: {q_value}"
        timer_color = (0, 255, 0) if self.lock_success_sent else FPS_TEXT_COLOR_BGR
        (timer_w, timer_h), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        (q_w, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        right_margin = 15
        timer_x = max(10, self.frame_w - timer_w - right_margin)
        timer_y = 62
        q_x = max(10, self.frame_w - q_w - right_margin)
        q_y = timer_y + max(timer_h, 22) + 10
        cv2.putText(frame, timer_text, (timer_x, timer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, timer_color, 2, cv2.LINE_AA)
        cv2.putText(frame, q_text, (q_x, q_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, FPS_TEXT_COLOR_BGR, 2, cv2.LINE_AA)

        # SPACE /target_data durumu yarismada karismasin diye ekranda HER ZAMAN gorunur.
        if manual_target_publish:
            if manual_is_fallback:
                space_text = f"SPACE /target_data: ON | FOCUS ID: {focused_target_id} | YOLO FALLBACK"
                space_color = MANUAL_FALLBACK_COLOR_BGR
            else:
                space_text = f"SPACE /target_data: ON | FOCUS ID: {focused_target_id}"
                space_color = MANUAL_FOCUSED_COLOR_BGR
        else:
            space_text = "SPACE /target_data: OFF | FOCUS: OFF"
            space_color = FPS_TEXT_COLOR_BGR

        (space_w, _), _ = cv2.getTextSize(space_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        space_x = max(10, self.frame_w - space_w - right_margin)
        space_y = q_y + 30
        cv2.putText(
            frame, space_text, (space_x, space_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, space_color, 2, cv2.LINE_AA,
        )

        current_server_dt = self.server_time_provider.now()
        draw_server_time(frame, current_server_dt, anchor="center", y=32)
        debug_draw_ms = (time.perf_counter() - debug_draw_start) * 1000.0
        frame_total_ms = (time.perf_counter() - frame_start) * 1000.0

        # /target_data = manuel FOCUS hedefi. LOCK timer artik track ID'ye bagli degildir;
        # payload'daki lock alanlari sistemin detection tabanli global lock durumunu gosterir.
        manual_is_lock_target = bool(manual_target_publish and manual_candidate is not None)
        target_data = {
            "frame": int(self.frame_id),
            "source_sequence": int(camera_fid),
            "timestamp": timestamp_iso,
            "target_id": focused_target_id,
            "tracking_status": "Tracked" if manual_candidate is not None else "Focused Lost",
            "target_valid": bool(manual_candidate is not None),
            "q_value": int(q_value),
            "queue_wait_ms": float(queue_wait_ms),
            "frame_age_ms": float(frame_age_ms),
            "confidence": float(manual_confidence),
            "normalized_error_x": float(manual_normalized_error_x),
            "normalized_error_y": float(manual_normalized_error_y),
            "target_velocity_x": float(manual_target_velocity_x),
            "target_velocity_y": float(manual_target_velocity_y),
            "bbox_area_ratio": float(manual_bbox_area_ratio),
            "center_x": int(manual_center_x),
            "center_y": int(manual_center_y),
            "bbox": [int(v) for v in manual_bbox_payload],
            "bbox_width_ratio": float(manual_width_ratio),
            "bbox_height_ratio": float(manual_height_ratio),
            "center_in_hit_area": bool(manual_center_in_hit_area),
            "size_ok": bool(manual_size_ok),
            "lock_elapsed_s": float(self.lock_elapsed_s if manual_is_lock_target else 0.0),
            "lock_required_s": float(self.lock_duration_s),
            "lock_success": bool(self.lock_success_sent and manual_is_lock_target),
            "tolerance_used_ms": int(tolerance_used_ms if manual_is_lock_target else 0),
            "yolo_track_ms": float(yolo_track_ms),
            "processing_ms": float(processing_ms),
            "fps": float(fps),
        }

        if manual_target_publish and manual_candidate is not None:
            publish_json(
                lock_ros_active,
                lock_target_pub,
                lock_target_topic,
                target_data,
                terminal_log=False,
            )
            self.manual_publish_frames += 1

        with lock_display_lock:
            latest_lock_display_frame = frame

        evaluation_written = self.evaluation_recorder.total_written_frames if self.evaluation_recorder else 0
        evaluation_duplicates = self.evaluation_recorder.duplicate_frames if self.evaluation_recorder else 0
        display_ms = 0.0

        lock_candidate_conf = float(candidate["confidence"]) if candidate is not None else 0.0
        lock_candidate_valid = bool(candidate is not None and candidate["valid_for_lock"])
        status_key = str(tracking_status)
        self.tracking_status_counts[status_key] = self.tracking_status_counts.get(status_key, 0) + 1

        self.csv_writer.writerow([
            self.frame_id, camera_fid, camera_fid, timestamp_iso,
            self.server_time_provider.source_name(), format_server_time(current_server_dt),
            current_target_id, status_key, bool(target_valid), len(evaluated_tracks),
            selected_target_id, focused_target_id, bool(manual_target_publish),
            q_value, queue_wait_ms, frame_age_ms,
            manual_confidence, manual_normalized_error_x, manual_normalized_error_y,
            manual_target_velocity_x, manual_target_velocity_y, manual_bbox_area_ratio,
            manual_center_x, manual_center_y, *manual_bbox_payload,
            manual_width_ratio, manual_height_ratio, manual_center_in_hit_area, manual_size_ok,
            self.lock_target_id, lock_candidate_conf, lock_candidate_valid,
            self.lock_elapsed_s, tolerance_used_ms, self.lock_success_sent,
            yolo_track_ms, processing_ms, fps,
            read_wait_ms, capture_pipe_read_ms, capture_interval_ms,
            ingest_queue_ms, evaluation_build_ms, evaluation_update_ms,
            debug_draw_ms, frame_total_ms, evaluation_written, evaluation_duplicates,
            self.backbone.restart_count if self.backbone else 0,
            vision_decoder_restart_count,
        ])
        if self.frame_id % 30 == 0:
            self.csv_file.flush()

        self.total_frames += 1
        self.frame_id += 1
        self.prev_time = now_wall

        self.latency_samples["read_wait_ms"].append(read_wait_ms)
        self.latency_samples["capture_pipe_read_ms"].append(capture_pipe_read_ms)
        if capture_interval_ms > 0:
            self.latency_samples["capture_interval_ms"].append(capture_interval_ms)
        self.latency_samples["ingest_queue_ms"].append(ingest_queue_ms)
        self.latency_samples["queue_wait_ms"].append(queue_wait_ms)
        self.latency_samples["yolo_track_ms"].append(float(yolo_track_ms))
        self.latency_samples["processing_ms"].append(processing_ms)
        self.latency_samples["frame_age_ms"].append(frame_age_ms)
        self.latency_samples["evaluation_build_ms"].append(evaluation_build_ms)
        self.latency_samples["evaluation_update_ms"].append(evaluation_update_ms)
        self.latency_samples["debug_draw_ms"].append(debug_draw_ms)
        self.latency_samples["frame_total_ms"].append(frame_total_ms)
        self.latency_samples["display_ms"].append(display_ms)
        self.latency_samples["q_value"].append(q_value)

    def stop(self):
        self.local_stop.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        self.write_report()

    def write_report(self):
        detection_ratio = self.detection_frames / self.total_frames * 100.0 if self.total_frames else 0.0
        valid_ratio = self.valid_candidate_frames / self.total_frames * 100.0 if self.total_frames else 0.0
        proc_stats = metric_stats(self.latency_samples["processing_ms"])
        avg_fps = 1000.0 / proc_stats["avg"] if proc_stats["avg"] > 0 else 0.0
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("TUNGA COMBINED SINGLE-RTSP YOLO + BYTETRACK LOCK REPORT\n")
            f.write("====================================================\n")
            f.write("Herelink RTSP client count from this program: 1\n")
            f.write("Mode architecture: Q=QR, K=YOLO+ByteTrack LOCK, mutually exclusive\n")
            f.write("K inference path: exactly 1x YoloByteTracker.track_with_detections(frame) per processed K frame\n")
            f.write("Separate second YOLO inference in K mode: NO\n")
            f.write(f"Total K processed frames: {self.total_frames}\n")
            f.write(f"Approx K FPS from avg processing time: {avg_fps:.2f}\n")
            f.write(f"Frames with raw YOLO detection: {self.detection_frames} ({detection_ratio:.2f}%)\n")
            f.write(f"Valid lock-candidate frames: {self.valid_candidate_frames} ({valid_ratio:.2f}%)\n")
            f.write(f"Successful locks: {self.lock_success_count}\n")
            f.write(f"Manual /target_data publish frames: {self.manual_publish_frames}\n")
            f.write(f"Manual focus toggle events: {self.manual_focus_events}\n")
            f.write(f"Tracking status counts: {json.dumps(self.tracking_status_counts, ensure_ascii=False)}\n")
            f.write(f"Lock duration: {self.lock_duration_s:.2f} s\n")
            f.write(f"Temporary YOLO-miss tolerance: {DEFAULT_LOCK_LOSS_TOLERANCE_S:.3f} s\n")
            f.write(f"Min target axis ratio: {self.min_axis_ratio:.4f}\n")
            f.write(f"Hit area pixels: {self.hit_area}\n")
            f.write("ByteTrack: ENABLED only for MPC manual target selection and /target_data; LOCK/timer independent\n")
            f.write("K application-level latest-frame skipping: DISABLED\n")
            f.write(f"Backbone restarts: {self.backbone.restart_count if self.backbone else 0}\n")
            f.write(f"Local vision decoder restarts: {vision_decoder_restart_count}\n\n")
            for key, values in self.latency_samples.items():
                write_metric_line(f, key, values, unit="frame" if key == "q_value" else "ms")




def get_lock_display_frame():
    with lock_display_lock:
        if latest_lock_display_frame is None:
            return None
        return latest_lock_display_frame.copy()


def build_combined_display():
    """Ana UI frame'i. GUI UDP bu pencere DEGIL; backbone raw H264 yayini kesintisizdir."""
    frame, frame_id = get_latest_frame_for_display()
    if frame is None:
        return None, frame_id

    mode = get_active_mode()
    if mode == MODE_LOCK:
        lock_frame = get_lock_display_frame()
        if lock_frame is not None:
            frame = lock_frame
    elif mode == MODE_QR:
        result, result_ts = get_last_qr_for_display()
        event_active = (
            result is not None
            and (time.monotonic() - result_ts) <= QR_DRAW_TTL_S
            and should_draw_result(result)
        )
        if DRAW_TARGET_AREA_ALWAYS:
            draw_target_area(frame)
        elif DRAW_TARGET_AREA_ON_EVENT and event_active:
            roi_color, _ = result_draw_color(result)
            draw_target_area(frame, color=roi_color)
        if event_active:
            draw_result(frame, result)
        frame = optical_zoom(frame, zoom_levels[zoom_index])
        draw_performance_overlay(frame)
        if DRAW_OVERLAY_ALWAYS:
            draw_overlay(frame)

    # Sadece local kontrol penceresine mode etiketi. GUI UDP/evaluation'a gitmez.
    cv2.putText(
        frame,
        f"MODE: {mode}",
        (15, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame, frame_id


def _read_mode_durations_from_system_csv(system_csv_path, total_duration_s):
    durations = {MODE_IDLE: 0.0, MODE_QR: 0.0, MODE_LOCK: 0.0}
    entries = {MODE_IDLE: 0, MODE_QR: 0, MODE_LOCK: 0}
    current_mode = MODE_IDLE
    last_t = 0.0
    entries[MODE_IDLE] = 1
    try:
        with open(system_csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("event") != "mode_change":
                    continue
                try:
                    t = float(row.get("session_elapsed_s", 0.0) or 0.0)
                except Exception:
                    continue
                durations[current_mode] = durations.get(current_mode, 0.0) + max(0.0, t - last_t)
                msg = str(row.get("message", ""))
                if "->" in msg:
                    rhs = msg.split("->", 1)[1].split(";", 1)[0].strip().upper()
                    if rhs in durations:
                        current_mode = rhs
                        entries[current_mode] = entries.get(current_mode, 0) + 1
                last_t = t
    except Exception:
        pass
    durations[current_mode] = durations.get(current_mode, 0.0) + max(0.0, float(total_duration_s) - last_t)
    return {k: round(v, 3) for k, v in durations.items()}, entries


def _read_runtime_metric_stats(runtime_csv_path):
    fields = [
        "camera_fps", "wechat_fps", "wechat_queue", "wechat_lag_frames",
        "wechat_delay_ms", "lock_queue",
    ]
    values = {k: [] for k in fields}
    mode_samples = {MODE_IDLE: 0, MODE_QR: 0, MODE_LOCK: 0}
    try:
        with open(runtime_csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                mode = str(row.get("mode", "")).upper()
                if mode in mode_samples:
                    mode_samples[mode] += 1
                for key in fields:
                    try:
                        values[key].append(float(row.get(key, 0.0) or 0.0))
                    except Exception:
                        pass
    except Exception:
        pass
    return {k: metric_stats(v) for k, v in values.items()}, mode_samples


def write_unified_flight_summary(
    logs_dir,
    flight_dir,
    lock_engine,
    backbone,
    evaluation_recorder,
    raw_video_path,
    evaluation_video_path,
):
    """K + Q + genel sistem + videolar için tek final summary.txt/json üretir."""
    logs_dir = Path(logs_dir)
    summary_json_path = logs_dir / "summary.json"
    summary_txt_path = logs_dir / "summary.txt"

    try:
        preliminary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    except Exception:
        preliminary = {}

    total_duration_s = float(preliminary.get("duration_s", 0.0) or 0.0)
    system_csv = logs_dir / "system_events.csv"
    runtime_csv = logs_dir / "runtime_metrics.csv"
    mode_durations, mode_entries = _read_mode_durations_from_system_csv(system_csv, total_duration_s)
    runtime_stats, runtime_mode_samples = _read_runtime_metric_stats(runtime_csv)

    lock_latency = {
        key: metric_stats(values)
        for key, values in (lock_engine.latency_samples.items() if lock_engine is not None else [])
    }
    lock_total = int(lock_engine.total_frames) if lock_engine is not None else 0
    lock_detection_frames = int(lock_engine.detection_frames) if lock_engine is not None else 0
    lock_valid_frames = int(lock_engine.valid_candidate_frames) if lock_engine is not None else 0

    final_summary = {
        "session_id": preliminary.get("session_id", ""),
        "start_utc": preliminary.get("start_utc", ""),
        "end_utc": _flight_utc_now(),
        "duration_s": round(total_duration_s, 3),
        "architecture": {
            "herelink_rtsp_clients": 1,
            "modes_mutually_exclusive": True,
            "keys": {"Q": "QR", "K": "LOCK", "ESC": "EXIT"},
        },
        "flight_directory": str(Path(flight_dir)),
        "mode_durations_s": mode_durations,
        "mode_entries": mode_entries,
        "camera": {
            "frames": int(preliminary.get("camera_frames", 0) or 0),
            "avg_fps": float(preliminary.get("camera_fps_avg", 0.0) or 0.0),
            "resolution": [int(W), int(H)],
        },
        "qr": {
            "workers": preliminary.get("qr_workers", {}),
            "dispatcher_actions": preliminary.get("qr_dispatcher_actions", {}),
            "publish_status_counts": preliminary.get("qr_publish_status_counts", {}),
            "profile": {
                "wechat": "every frame FIFO, single pass, target ROI + margin",
                "target_scan_margin_px": int(TARGET_SCAN_MARGIN_PX),
                "pyzbar_every_n": int(PYZBAR_WORKER_EVERY_N),
                "opencv_worker": "disabled",
                "enhanced_worker": "disabled",
                "mpc": "disabled",
            },
        },
        "lock": {
            "processed_frames": lock_total,
            "detection_frames": lock_detection_frames,
            "detection_ratio_pct": round(lock_detection_frames / lock_total * 100.0, 3) if lock_total else 0.0,
            "valid_candidate_frames": lock_valid_frames,
            "valid_candidate_ratio_pct": round(lock_valid_frames / lock_total * 100.0, 3) if lock_total else 0.0,
            "successful_locks": int(lock_engine.lock_success_count) if lock_engine is not None else 0,
            "lock_duration_s": float(lock_engine.lock_duration_s) if lock_engine is not None else 0.0,
            "loss_tolerance_s": float(DEFAULT_LOCK_LOSS_TOLERANCE_S),
            "tracker": "ByteTrack",
            "tracking_status_counts": dict(getattr(lock_engine, "tracking_status_counts", {})) if lock_engine is not None else {},
            "manual_target_publish_frames": int(getattr(lock_engine, "manual_publish_frames", 0)) if lock_engine is not None else 0,
            "manual_focus_events": int(getattr(lock_engine, "manual_focus_events", 0)) if lock_engine is not None else 0,
            "latency_metrics": lock_latency,
        },
        "general_runtime": {
            "metrics": runtime_stats,
            "samples_by_mode": runtime_mode_samples,
            "log_dropped_events": int(preliminary.get("log_dropped_events", 0) or 0),
        },
        "video_pipeline": {
            "backbone_restarts": int(backbone.restart_count) if backbone is not None else 0,
            "vision_decoder_restarts": int(vision_decoder_restart_count),
            "evaluation_written_frames": int(evaluation_recorder.total_written_frames) if evaluation_recorder is not None else 0,
            "evaluation_duplicate_frames": int(evaluation_recorder.duplicate_frames) if evaluation_recorder is not None else 0,
            "evaluation_deadline_misses": int(evaluation_recorder.deadline_misses) if evaluation_recorder is not None else 0,
            "evaluation_broken_pipe": bool(evaluation_recorder.broken_pipe) if evaluation_recorder is not None else False,
        },
        "files": {
            "raw_video": str(raw_video_path) if raw_video_path is not None else None,
            "evaluation_video": str(evaluation_video_path) if evaluation_video_path is not None else None,
            "logs_dir": str(logs_dir),
            "runtime_metrics": str(logs_dir / "runtime_metrics.csv"),
            "lock_frames": str(logs_dir / "lock_frames.csv"),
            "lock_report": str(logs_dir / "lock_report.txt"),
            "qr_worker_attempts": str(logs_dir / "qr_worker_attempts.csv"),
            "qr_detections": str(logs_dir / "qr_detections.csv"),
            "system_events": str(logs_dir / "system_events.csv"),
            "ffmpeg_backbone": str(logs_dir / "ffmpeg_backbone.log"),
            "ffmpeg_evaluation": str(logs_dir / "ffmpeg_evaluation.log"),
        },
    }

    summary_json_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "TUNGA - UNIFIED FLIGHT SUMMARY",
        "=" * 64,
        f"Flight dir: {flight_dir}",
        f"Duration: {final_summary['duration_s']:.3f} s",
        "Herelink RTSP client: 1",
        "",
        "MODE SURELERI",
        f"  IDLE: {mode_durations.get(MODE_IDLE, 0.0):.3f} s",
        f"  QR:   {mode_durations.get(MODE_QR, 0.0):.3f} s",
        f"  LOCK: {mode_durations.get(MODE_LOCK, 0.0):.3f} s",
        "",
        "GENEL",
        f"  Camera frames: {final_summary['camera']['frames']}",
        f"  Camera avg FPS: {final_summary['camera']['avg_fps']:.3f}",
        f"  Backbone restart: {final_summary['video_pipeline']['backbone_restarts']}",
        f"  Local decoder restart: {final_summary['video_pipeline']['vision_decoder_restarts']}",
        f"  Evaluation written/duplicate/deadline-miss: "
        f"{final_summary['video_pipeline']['evaluation_written_frames']} / "
        f"{final_summary['video_pipeline']['evaluation_duplicate_frames']} / "
        f"{final_summary['video_pipeline']['evaluation_deadline_misses']}",
        "",
        "QR",
        f"  Publish status: {json.dumps(final_summary['qr']['publish_status_counts'], ensure_ascii=False)}",
    ]
    for worker, stat in final_summary["qr"]["workers"].items():
        lines.append(
            f"  {worker}: processed={stat.get('processed_frames', 0)}, "
            f"readable={stat.get('readable_detections', 0)}, "
            f"proc_avg={stat.get('processing_ms_avg', 0):.3f}ms, "
            f"capture_finish_avg={stat.get('capture_to_finish_ms_avg', 0):.3f}ms"
        )

    lines.extend([
        "",
        "LOCK",
        f"  Processed frames: {lock_total}",
        f"  Detection frames: {lock_detection_frames}",
        f"  Valid candidate frames: {lock_valid_frames}",
        f"  Successful locks: {final_summary['lock']['successful_locks']}",
        f"  Tracker: {final_summary['lock']['tracker']}",
        f"  Tracking status: {json.dumps(final_summary['lock']['tracking_status_counts'], ensure_ascii=False)}",
        f"  Manual /target_data publish frames: {final_summary['lock']['manual_target_publish_frames']}",
        f"  Manual focus events: {final_summary['lock']['manual_focus_events']}",
        "",
        "LOCK GECIKME / QUEUE METRIKLERI",
    ])
    for key, stat in lock_latency.items():
        unit = "frame" if key == "q_value" else "ms"
        lines.append(
            f"  {key}: avg={stat['avg']:.3f} {unit}, p50={stat['p50']:.3f}, "
            f"p95={stat['p95']:.3f}, max={stat['max']:.3f}, n={stat['count']}"
        )

    lines.extend([
        "",
        "GENEL RUNTIME METRIKLERI",
    ])
    for key, stat in runtime_stats.items():
        lines.append(
            f"  {key}: avg={stat['avg']:.3f}, p50={stat['p50']:.3f}, "
            f"p95={stat['p95']:.3f}, max={stat['max']:.3f}, n={stat['count']}"
        )

    lines.extend([
        "",
        "DOSYALAR",
        f"  Raw: {raw_video_path}",
        f"  Evaluation: {evaluation_video_path}",
        f"  Logs: {logs_dir}",
    ])
    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LOG] FINAL unified summary: {summary_txt_path}")


def combined_main():
    global W, H, RTSP_URL, INPUT_MODE, TOP_CROP_PX
    global lock_frame_queue, shared_evaluation_recorder, shared_server_time_provider
    global shared_backbone, vision_relay_host, vision_relay_port, zoom_index

    # Sartname video adlandirmasi: program acilir acilmaz musabaka numarasini iste.
    competition_number = prompt_competition_number()

    # LOCK config mevcut dosyadaki config.yaml ile ayni.
    cfg = load_config("config.yaml")
    outputs_dir = Path(ensure_dir(cfg["outputs_dir"]))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flight_dir = outputs_dir / f"flight_{run_stamp}"
    logs_dir = flight_dir / "logs"
    flight_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Birlesik kodda tum loglar zorunlu olarak bu tek klasore gider.
    FLIGHT_LOGS_DIR = str(logs_dir)

    # Tek ortak processing boyutu. Varsayilan iki kodda da 1280x720.
    W = int(cfg.get("processing_width", cfg.get("capture_width", DEFAULT_PROCESS_WIDTH)))
    H = int(cfg.get("processing_height", cfg.get("capture_height", DEFAULT_PROCESS_HEIGHT)))
    TOP_CROP_PX = max(0, int(cfg.get("top_crop_px", DEFAULT_TOP_CROP_PX)))
    if TOP_CROP_PX >= H:
        raise RuntimeError(
            f"top_crop_px gecersiz: {TOP_CROP_PX}. Processing yuksekliginden ({H}) kucuk olmali."
        )
    RTSP_URL = str(cfg.get("rtsp_url", DEFAULT_RTSP_URL)).strip()
    INPUT_MODE = "rtsp"  # Combined competition omurgasi Herelink RTSP icindir.

    # QR'nin ayrintili ROS parametrelerini ve /qr_data publisher'ini kurar.
    init_ros_if_available()
    # QR ROS param capture override varsa W/H burada guncellenmis olabilir.
    if USE_ROS and rospy is not None:
        RTSP_URL = rospy.get_param("~rtsp_url", RTSP_URL)

    # Sartname guvenligi: minimum cozunurluk + izin verilen aspect ratio zorunlu.
    validate_competition_resolution(W, H)

    # /server_time ZORUNLU. Local PC saati fallback olarak kullanilmaz.
    if not USE_ROS or rospy is None or String is None:
        raise RuntimeError(
            "[FATAL] /server_time zorunlu ancak ROS/std_msgs String kullanilamiyor. "
            "Program baslatilmadi."
        )

    shared_server_time_provider = ServerTimeProvider()
    server_time_sub = None
    try:
        server_time_sub = rospy.Subscriber(
            DEFAULT_SERVER_TIME_TOPIC,
            String,
            shared_server_time_provider.server_time_callback,
            queue_size=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[FATAL] {DEFAULT_SERVER_TIME_TOPIC} subscriber acilamadi: {exc}"
        ) from exc

    server_time_timeout_s = max(0.1, float(cfg.get(
        "server_time_startup_timeout_s", DEFAULT_SERVER_TIME_STARTUP_TIMEOUT_S
    )))
    print(
        f"[INFO] {DEFAULT_SERVER_TIME_TOPIC} bekleniyor "
        f"(timeout={server_time_timeout_s:.1f}s)..."
    )
    if not shared_server_time_provider.wait_until_ready(server_time_timeout_s):
        try:
            server_time_sub.unregister()
        except Exception:
            pass
        raise RuntimeError(
            f"[FATAL] {DEFAULT_SERVER_TIME_TOPIC} uzerinden gecerli sunucu saati "
            f"{server_time_timeout_s:.1f} saniye icinde gelmedi. Program baslatilmadi."
        )

    print(
        f"[INFO] Sunucu saati hazir: "
        f"{format_server_time(shared_server_time_provider.now())} "
        f"| kaynak={shared_server_time_provider.source_name()}"
    )

    init_lock_ros_after_qr(cfg)

    try:
        cv2.setNumThreads(max(1, int(OPENCV_INTERNAL_THREADS)))
    except Exception:
        pass

    # K modu: tek YOLO inference. Ham detection LOCK/timer icin; ByteTrack sadece MPC /target_data icin.
    # Tracker/model program basinda bir kez yuklenir; Q/K gecisleri Herelink RTSP'yi etkilemez.
    print(f"[INFO] YOLO + ByteTrack bir kez yukleniyor: {cfg['weights']}")
    yolo_tracker = YoloByteTracker(
        weights=cfg["weights"],
        conf_threshold=float(cfg["conf_threshold"]),
        iou_threshold=float(cfg["iou_threshold"]),
        imgsz=int(cfg["imgsz"]),
        device=cfg["device"],
    )

    # Tek uçuş / tek logs klasörü. QR + LOCK + genel sistem burada toplanır.
    reset_runtime_buffers()
    stop_event.clear()
    flight_log_close_event.clear()
    FLIGHT_LOGS_DIR = str(logs_dir)
    initialize_flight_log_session(session_dir=logs_dir, session_id=run_stamp)

    # Mode IDLE ile baslar; yayin ve kayit hemen baslar.
    set_active_mode(MODE_IDLE)

    raw_record_enabled = parse_bool(cfg.get("raw_record_enabled", DEFAULT_RAW_RECORD_ENABLED))
    gui_stream_enabled = parse_bool(cfg.get("gui_stream_enabled", DEFAULT_GUI_STREAM_ENABLED))
    evaluation_record_enabled = parse_bool(cfg.get("evaluation_record_enabled", DEFAULT_EVALUATION_RECORD_ENABLED))
    gui_stream_host = str(cfg.get("gui_stream_host", DEFAULT_GUI_STREAM_HOST)).strip()
    gui_stream_port = int(cfg.get("gui_stream_port", DEFAULT_GUI_STREAM_PORT))
    evaluation_fps = max(15.0, float(cfg.get("evaluation_fps", DEFAULT_EVALUATION_FPS)))
    evaluation_encoder = str(cfg.get("evaluation_encoder", DEFAULT_EVALUATION_ENCODER)).strip()
    evaluation_preset = str(cfg.get("evaluation_preset", DEFAULT_EVALUATION_PRESET)).strip()
    evaluation_crf = int(cfg.get("evaluation_crf", DEFAULT_EVALUATION_CRF))
    # Yeni K kodundaki frame_queue_size varsa onu kullan; yoksa eski combined reader_queue_size.
    reader_queue_size = max(1, int(cfg.get("frame_queue_size", cfg.get("reader_queue_size", DEFAULT_READER_QUEUE_SIZE))))
    vision_relay_host = str(cfg.get("vision_relay_host", DEFAULT_VISION_RELAY_HOST)).strip()
    vision_relay_port = int(cfg.get("vision_relay_port", DEFAULT_VISION_RELAY_PORT))

    lock_frame_queue = queue.Queue(maxsize=reader_queue_size)

    # Her uçuş klasöründe normal durumda sadece 2 video + logs/ vardır.
    raw_video_path = flight_dir / "raw_720p.mkv" if raw_record_enabled else None
    evaluation_video_path = flight_dir / build_competition_video_filename(competition_number)
    backbone_log_path = logs_dir / "ffmpeg_backbone.log"
    evaluation_log_path = logs_dir / "ffmpeg_evaluation.log"
    lock_csv_path = logs_dir / "lock_frames.csv"
    lock_report_path = logs_dir / "lock_report.txt"

    shared_evaluation_recorder = None
    if evaluation_record_enabled:
        shared_evaluation_recorder = EvaluationRecorder(
            output_path=evaluation_video_path,
            width=W,
            height=H,
            fps=evaluation_fps,
            time_provider=shared_server_time_provider,
            ffmpeg_log_path=evaluation_log_path,
            encoder=evaluation_encoder,
            preset=evaluation_preset,
            crf=evaluation_crf,
        )
        shared_evaluation_recorder.start()

    shared_backbone = PersistentVideoBackbone(
        rtsp_url=RTSP_URL,
        raw_base_path=raw_video_path,
        gui_enabled=gui_stream_enabled,
        gui_host=gui_stream_host,
        gui_port=gui_stream_port,
        relay_host=vision_relay_host,
        relay_port=vision_relay_port,
        ffmpeg_log_path=backbone_log_path,
    )
    shared_backbone.start()

    # Local decoder backbone UDP output'u acildiktan sonra baslasin.
    time.sleep(0.15)

    lock_engine = LockEngine(
        cfg=cfg,
        tracker=yolo_tracker,
        evaluation_recorder=shared_evaluation_recorder,
        server_time_provider=shared_server_time_provider,
        csv_path=lock_csv_path,
        report_path=lock_report_path,
        backbone=shared_backbone,
    )

    log_thread = threading.Thread(name="flight_logger", target=flight_logger_thread_func, daemon=True)
    cam_thread = threading.Thread(name="shared_camera", target=camera_thread_func, daemon=True)
    result_thread = threading.Thread(name="result_dispatcher", target=result_dispatcher_thread_func, daemon=True)
    worker_threads = [
        # OpenCV worker DEVRE DISI - eski fonksiyon kodda duruyor.
        threading.Thread(name="wechat_worker", target=wechat_worker_thread_func, daemon=True),
        threading.Thread(name="pyzbar_worker", target=pyzbar_worker_thread_func, daemon=True),
        # Enhanced worker DEVRE DISI - eski fonksiyon kodda duruyor.
    ]

    log_thread.start()
    log_system_event("program_start", "combined_single_rtsp_qr_lock")
    result_thread.start()
    cam_thread.start()
    for t in worker_threads:
        t.start()
    lock_engine.start()

    show_window = parse_bool(cfg.get("show", True))
    if show_window:
        cv2.namedWindow(COMBINED_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        window_width = max(320, int(cfg.get("window_width", 960)))
        window_height = max(240, int(cfg.get("window_height", 540)))
        cv2.resizeWindow(COMBINED_WINDOW_TITLE, window_width, window_height)

    print("\n============================================================")
    print("TUNGA COMBINED SINGLE-RTSP SISTEM AKTIF")
    print("============================================================")
    print(f"Flight klasoru: {flight_dir}")
    print(f"Tek logs klasoru: {logs_dir}")
    print(f"Herelink RTSP: {RTSP_URL}")
    print("HERELINK RTSP CLIENT SAYISI: 1")
    print(f"Raw kayit: {raw_video_path if raw_record_enabled else 'KAPALI'}")
    print(f"GUI: udp://{gui_stream_host}:{gui_stream_port} (H264 copy)" if gui_stream_enabled else "GUI: KAPALI")
    print(f"Evaluation: {evaluation_video_path if evaluation_record_enabled else 'KAPALI'}")
    print(f"Local vision relay: udp://{vision_relay_host}:{vision_relay_port}")
    print(f"Processing: {W}x{H} | top crop: {TOP_CROP_PX}px (crop -> scale -> tum vision islemleri)")
    print(f"LOCK queue max: {reader_queue_size}; application-level drop YOK | YOLO LOCK + ByteTrack MPC")
    print(f"QR: WeChat HER FRAME FIFO/TEK OKUMA, vurus alani + {TARGET_SCAN_MARGIN_PX}px; pyzbar every_n={PYZBAR_WORKER_EVERY_N}")
    print("QR OpenCV worker: DEVRE DISI")
    print("QR Enhanced worker: DEVRE DISI")
    print("QR MPC: DEVRE DISI")
    print("QR raw recording: DEVRE DISI (ortak raw kayit backbone'da)")
    print("TUSLAR: Q=QR | K=LOCK+TRACK | B=sol hedef | N=sag hedef | SPACE=focus | -=timer reset | Z=QR zoom | ESC=cikis")
    print("Q/K gecisleri RTSP/GUI/raw/evaluation proseslerini yeniden BASLATMAZ.")
    print("============================================================\n")

    try:
        while not stop_event.is_set() and not ros_is_shutdown():
            if show_window:
                display_frame, _ = build_combined_display()
                if display_frame is not None:
                    cv2.imshow(COMBINED_WINDOW_TITLE, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    set_active_mode(MODE_QR)
                elif key in (ord('k'), ord('K')):
                    set_active_mode(MODE_LOCK)
                elif key in (ord('z'), ord('Z')):
                    if get_active_mode() == MODE_QR:
                        zoom_index = (zoom_index + 1) % len(zoom_levels)
                        print(f"[INFO] QR zoom: {zoom_levels[zoom_index]}x")
                elif get_active_mode() == MODE_LOCK and (key in (ord('b'), ord('B'), ord('n'), ord('N'), ord('-')) or key == 32):
                    lock_engine.handle_key(key)
                elif key == 27:  # ESC
                    stop_event.set()
                    break
            else:
                # Klavye mode switch OpenCV penceresi uzerinden yapildigi icin show=False testinde
                # mod secimi ROS/terminal entegrasyonuna birakilir. Yarismada show=True kullan.
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C alindi.")
        stop_event.set()

    finally:
        # Once processing worker'larini durdur. BACKBONE en son kapanir ki GUI en uzun sure acik kalsin.
        stop_event.set()
        with frame_condition:
            frame_condition.notify_all()
        _drain_queue(lock_frame_queue)

        try:
            cam_thread.join(timeout=3.0)
        except Exception:
            pass
        for t in worker_threads:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        try:
            result_thread.join(timeout=2.0)
        except Exception:
            pass

        lock_engine.stop()

        log_system_event("program_stop", "combined_single_rtsp_qr_lock", frame_id=latest_frame_id)
        flight_log_close_event.set()
        try:
            log_thread.join(timeout=8.0)
        except Exception:
            pass

        # Evaluation'i backbone'dan once finalize et.
        if shared_evaluation_recorder is not None:
            shared_evaluation_recorder.stop()

        # GUI/raw omurgasi EN SON kapanir.
        if shared_backbone is not None:
            shared_backbone.stop()

        # QR preliminary summary + LOCK + genel runtime + video pipeline -> TEK final summary.
        try:
            write_unified_flight_summary(
                logs_dir=logs_dir,
                flight_dir=flight_dir,
                lock_engine=lock_engine,
                backbone=shared_backbone,
                evaluation_recorder=shared_evaluation_recorder,
                raw_video_path=raw_video_path,
                evaluation_video_path=evaluation_video_path if evaluation_record_enabled else None,
            )
        except Exception as exc:
            print(f"[WARN] Unified summary yazilamadi: {exc}")

        cv2.destroyAllWindows()
        if USE_ROS and rospy is not None:
            try:
                rospy.signal_shutdown("combined single-rtsp vision finished")
            except Exception:
                pass

    print("\n[INFO] Sistem kapandi.")
    print(f"Backbone restart: {shared_backbone.restart_count if shared_backbone else 0}")
    print(f"Local vision decoder restart: {vision_decoder_restart_count}")
    print(f"Flight klasoru: {flight_dir}")
    print(f"Tum loglar: {logs_dir}")
    print(f"Unified summary: {logs_dir / 'summary.txt'}")
    if shared_backbone and shared_backbone.raw_paths:
        print("Raw kayit(lar):")
        for p in shared_backbone.raw_paths:
            print(f"  - {p}")
    if evaluation_record_enabled:
        print(f"Evaluation video: {evaluation_video_path}")


if __name__ == "__main__":
    combined_main()
