#!/usr/bin/env python3
"""Small environment check for the public repository."""

import importlib
import shutil
import sys


def check_import(name, display=None):
    display = display or name
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        suffix = f" ({version})" if version else ""
        print(f"[OK]   {display}{suffix}")
        return module
    except Exception as exc:
        print(f"[FAIL] {display}: {exc}")
        return None


print(f"Python: {sys.version.split()[0]}")
print("\nPython packages")
cv2 = check_import("cv2", "OpenCV")
check_import("numpy", "NumPy")
check_import("yaml", "PyYAML")
check_import("ultralytics", "Ultralytics")
check_import("pyzbar.pyzbar", "pyzbar")

print("\nExternal tools")
for tool in ("ffmpeg", "ffprobe"):
    path = shutil.which(tool)
    print(f"[OK]   {tool}: {path}" if path else f"[FAIL] {tool}: not found")

print("\nROS")
rospy = check_import("rospy", "rospy")
check_import("std_msgs.msg", "std_msgs")

if cv2 is not None:
    wechat = hasattr(cv2, "wechat_qrcode_WeChatQRCode") or (
        hasattr(cv2, "wechat_qrcode") and hasattr(cv2.wechat_qrcode, "WeChatQRCode")
    )
    print(f"\nWeChatQRCode support: {'YES' if wechat else 'NO'}")
