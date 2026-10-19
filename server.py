"""
Handy - FastAPI backend

What this file does:
  1. Opens your webcam ONCE, in a background thread.
  2. Runs every frame through your Handtracker (the same class you already wrote).
  3. Streams the annotated frames to the browser as a live video (MJPEG).
  4. Offers small "API" endpoints the web page calls: capture, reset, list captures.

Run it with:   python server.py
Then open:     http://127.0.0.1:8000
"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from Handtracker import Handtracker

BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "static" / "index.html"


class CameraWorker:
    """Reads the webcam in a background thread so the web server never blocks."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.tracker = Handtracker()

        self.cap = None
        self.thread = None
        self.running = False

        # A lock stops two threads (camera loop + web requests) from touching
        # the same data at the same moment.
        self.lock = threading.Lock()

        self.latest_jpeg = None   # newest annotated frame, ready to stream
        self.clean_frame = None   # newest frame WITHOUT drawings (used for captures)

        self.captures = []        # every capture taken so far (kept in memory)
        self.next_id = 1

    # ---------- start / stop ----------
    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("!! Could not open the camera. Close other apps using it and restart.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("Camera started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    # ---------- the camera loop (runs forever in its own thread) ----------
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            clean = frame.copy()  # untouched copy, so captures don't include the green square

            with self.lock:
                annotated = self.tracker.frame_process(frame)
                self.clean_frame = clean

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()

    # ---------- things the web page can ask for ----------
    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def capture(self):
        """Crop the drawn square. Returns the new capture, or None if there's no square."""
        with self.lock:
            if self.clean_frame is None:
                return None
            coords, crop = self.tracker.capture(self.clean_frame)

        if crop is None:
            return None

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None

        x1, y1, x2, y2 = coords
        with self.lock:
            item = {
                "id": self.next_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "time": time.strftime("%H:%M:%S"),
                "jpeg": buf.tobytes(),
            }
            self.next_id += 1
            self.captures.append(item)
        return item

    def reset_square(self):
        with self.lock:
            self.tracker.reset_rectangle()

    def clear_captures(self):
        with self.lock:
            self.captures.clear()

    def status(self):
        with self.lock:
            return {
                "camera_ok": self.running and self.latest_jpeg is not None,
                "has_square": self.tracker.drawn_square is not None,
                "capture_count": len(self.captures),
            }


def public(item):
    """The capture info we send to the browser (without the raw image bytes)."""
    return {k: v for k, v in item.items() if k != "jpeg"}


camera = CameraWorker()


@asynccontextmanager
async def lifespan(app):
    camera.start()   # runs when the server starts
    yield
    camera.stop()    # runs when the server shuts down


app = FastAPI(title="Handy", lifespan=lifespan)


# ---------- the web page ----------
@app.get("/")
def home():
    return FileResponse(INDEX_HTML)


# ---------- live video ----------
def mjpeg_frames():
    """Yields the newest frame again and again. The browser shows it as a video."""
    while True:
        jpeg = camera.get_jpeg()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.03)  # about 30 frames per second


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------- API ----------
@app.get("/api/status")
def status():
    return camera.status()


@app.post("/api/capture")
def capture():
    item = camera.capture()
    if item is None:
        raise HTTPException(status_code=400, detail="No square yet. Pinch with both hands to draw one.")
    return public(item)


@app.post("/api/reset")
def reset():
    camera.reset_square()
    return {"ok": True}


@app.get("/api/captures")
def list_captures():
    with camera.lock:
        items = [public(c) for c in camera.captures]
    return list(reversed(items))  # newest first


@app.get("/api/captures/{capture_id}/image")
def capture_image(capture_id: int):
    with camera.lock:
        for c in camera.captures:
            if c["id"] == capture_id:
                return Response(content=c["jpeg"], media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Capture not found")


@app.delete("/api/captures")
def clear_captures():
    camera.clear_captures()
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 