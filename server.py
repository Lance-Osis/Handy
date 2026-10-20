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

# ---------- settings ----------
BASE_DIR = Path(__file__).parent
PAGE = BASE_DIR / "static" / "web.html"
CERT_FILE = BASE_DIR / "cert.pem"
KEY_FILE = BASE_DIR / "key.pem"
PORT = 8443

STREAM_JPEG_QUALITY = 75     # frames sent back to the phone (speed matters)
CAPTURE_JPEG_QUALITY = 95    # saved captures (quality matters)

NO_SQUARE_MESSAGE = "No square yet. Pinch with both hands to draw one."

app = FastAPI(title="Handy phone mode")


@app.get("/")
def home():
    return FileResponse(PAGE)


# ---------- frame processing ----------
def encode_jpeg(image, quality):
    """Return the image as JPEG bytes, or None if encoding failed."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def process_frame(tracker, jpeg_bytes):
    """Decode one JPEG from the phone, run the tracker, return the results.
    (Runs in a worker thread so the server stays responsive.)

    Returns (clean_frame, annotated_jpeg, has_square), or None for a bad frame.
    """
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None

    frame = cv2.flip(frame, 1)   # mirror view, same as your PC version
    clean = frame.copy()         # untouched copy for captures
    annotated = tracker.frame_process(frame)

    annotated_jpeg = encode_jpeg(annotated, STREAM_JPEG_QUALITY)
    if annotated_jpeg is None:
        return None
    return clean, annotated_jpeg, tracker.drawn_square is not None


# ---------- one connected phone ----------
class PhoneSession:
    """Everything that belongs to ONE phone: its tracker, newest frame and capture counter.
    Every phone gets its own tracker, so each has its own square."""

    def __init__(self, ws):
        self.ws = ws
        self.tracker = Handtracker()
        self.clean = None    # newest un-annotated frame of this phone
        self.next_id = 1

    async def on_frame(self, jpeg_bytes):
        """A camera frame arrived: track it and send the annotated version back."""
        result = await asyncio.to_thread(process_frame, self.tracker, jpeg_bytes)
        if result is None:
            await self.ws.send_json({"type": "skip"})   # bad frame, ask for the next one
            return

        self.clean, annotated_jpeg, has_square = result
        await self.ws.send_json({"type": "status", "has_square": has_square})
        await self.ws.send_bytes(annotated_jpeg)

    async def on_command(self, command):
        """A button press arrived."""
        if command == "capture":
            await self.capture()
        elif command == "reset":
            self.tracker.reset_rectangle()
            await self.ws.send_json({"type": "status", "has_square": False})

    async def capture(self):
        coords, crop = (None, None)
        if self.clean is not None:
            coords, crop = self.tracker.capture(self.clean)
        if crop is None:
            await self.ws.send_json({"type": "error", "detail": NO_SQUARE_MESSAGE})
            return

        x1, y1, x2, y2 = coords
        crop_jpeg = encode_jpeg(crop, CAPTURE_JPEG_QUALITY)
        await self.ws.send_json({
            "type": "capture",
            "id": self.next_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "frame_width": self.clean.shape[1], "frame_height": self.clean.shape[0],
            "time": time.strftime("%H:%M:%S"),
            "image": base64.b64encode(crop_jpeg).decode("ascii"),
        })
        self.next_id += 1

    def close(self):
        try:
            self.tracker.hands.close()   # free the MediaPipe resources
        except Exception:
            pass


@app.websocket("/ws")
async def phone_socket(ws: WebSocket):
    await ws.accept()
    session = PhoneSession(ws)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                await session.on_frame(msg["bytes"])
            elif msg.get("text") is not None:
                await session.on_command(msg["text"])
    except WebSocketDisconnect:
        pass
    finally:
        session.close()


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


def cert_covers_ip(ip):
    """True if the existing cert.pem/key.pem already work for this IP."""
    from cryptography import x509

    if not (CERT_FILE.exists() and KEY_FILE.exists()):
        return False
    try:
        cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        return ipaddress.ip_address(ip) in san.get_values_for_type(x509.IPAddress)
    except Exception:
        return False


def create_cert(ip):
    """Write a new self-signed cert.pem/key.pem valid for localhost and this IP."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Handy local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    ips = {ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address(ip)}
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


def ensure_cert(ip):
    """Create cert.pem/key.pem if missing, or if your PC's IP changed."""
    try:
        import cryptography  # noqa: F401  (only checking it is installed)
    except ImportError:
        raise SystemExit("Missing package. Run:  pip install cryptography websockets")

    if cert_covers_ip(ip):
        return
    print("Creating a self-signed certificate...")
    create_cert(ip)


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
