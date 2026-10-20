"""
Handy - MOBILE MODE 

Here the MOBILE films you and this PC does the hand tracking:

  mobile camera -> (WebSocket, one JPEG per frame) -> this server runs Handtracker
  mobile screen <- (WebSocket, annotated JPEG back) <-

Why HTTPS? Phone browsers only allow camera access on secure pages (https://).
This script makes a self-signed certificate for you automatically the first time
(cert.pem + key.pem). Your phone will warn that it isn't trusted. That's expected,
because you made the certificate yourself. Proceed anyway.

Run it with:   python server.py
"""
import asyncio
import base64
import datetime
import ipaddress
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from handtracker import Handtracker

BASE_DIR = Path(__file__).parent
PAGE = BASE_DIR / "static" / "web.html"
CERT_FILE = BASE_DIR / "cert.pem"
KEY_FILE = BASE_DIR / "key.pem"
PORT = 8443

app = FastAPI(title="Handy phone mode")


@app.get("/")
def home():
    return FileResponse(PAGE)


def process_frame(tracker, jpeg_bytes):
    """Decode one JPEG from the phone, run the tracker, return the results.
    (Runs in a worker thread so the server stays responsive.)"""
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None

    frame = cv2.flip(frame, 1)   # mirror view, same as your PC version
    clean = frame.copy()         # untouched copy for captures
    annotated = tracker.frame_process(frame)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    return clean, buf.tobytes(), tracker.drawn_square is not None


@app.websocket("/ws")
async def phone_socket(ws: WebSocket):
    await ws.accept()

    # Every connected phone gets its OWN tracker, so each has its own square.
    tracker = Handtracker()
    clean = None          # newest un-annotated frame of this phone
    next_id = 1

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # ---- a camera frame from the phone ----
            if msg.get("bytes") is not None:
                result = await asyncio.to_thread(process_frame, tracker, msg["bytes"])
                if result is None:
                    await ws.send_json({"type": "skip"})   # bad frame, ask for the next one
                    continue
                clean, annotated_jpeg, has_square = result
                await ws.send_json({"type": "status", "has_square": has_square})
                await ws.send_bytes(annotated_jpeg)

            # ---- a button press ----
            elif msg.get("text") is not None:
                command = msg["text"]

                if command == "capture":
                    coords, crop = (None, None)
                    if clean is not None:
                        coords, crop = tracker.capture(clean)
                    if crop is None:
                        await ws.send_json({"type": "error",
                                            "detail": "No square yet. Pinch with both hands to draw one."})
                        continue

                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    x1, y1, x2, y2 = coords
                    await ws.send_json({
                        "type": "capture",
                        "id": next_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": x2 - x1, "height": y2 - y1,
                        "frame_width": clean.shape[1], "frame_height": clean.shape[0],
                        "time": time.strftime("%H:%M:%S"),
                        "image": base64.b64encode(buf.tobytes()).decode("ascii"),
                    })
                    next_id += 1

                elif command == "reset":
                    tracker.reset_rectangle()
                    await ws.send_json({"type": "status", "has_square": False})

    except WebSocketDisconnect:
        pass
    finally:
        try:
            tracker.hands.close()   # free the MediaPipe resources
        except Exception:
            pass


# ---------- HTTPS certificate helpers ----------
def get_lan_ip():
    """The PC's address on your Wi-Fi (what the phone types in)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # no data is actually sent
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip):
    """Create cert.pem/key.pem if missing, or if your PC's IP changed."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise SystemExit("Missing package. Run:  pip install cryptography websockets")

    wanted = ipaddress.ip_address(ip)

    if CERT_FILE.exists() and KEY_FILE.exists():
        try:
            cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            if wanted in san.get_values_for_type(x509.IPAddress):
                return   # existing certificate already covers this IP
        except Exception:
            pass

    print("Creating a self-signed certificate...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Handy local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    ips = {ipaddress.ip_address("127.0.0.1"), wanted}
    alt_names = [x509.DNSName("localhost")] + [x509.IPAddress(i) for i in ips]

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def main():
    ip = get_lan_ip()
    ensure_cert(ip)

    print()
    print("Phone mode is starting.")
    print(f"  On your phone (same Wi-Fi), open:  https://{ip}:{PORT}")
    print(f"  Testing on this PC instead?        https://localhost:{PORT}")
    print("  Your browser will warn the certificate is not trusted.")
    print("  That's expected. Choose Advanced -> Proceed.")
    print()

    uvicorn.run(app, host="0.0.0.0", port=PORT,
                ssl_keyfile=str(KEY_FILE), ssl_certfile=str(CERT_FILE))


if __name__ == "__main__":
    main()
