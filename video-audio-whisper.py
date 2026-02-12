from flask import Flask, request, jsonify
import os
import tempfile
import requests
import traceback
import cv2
import threading
import subprocess
from supabase import create_client, Client
from dotenv import load_dotenv

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Environment ----------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_TRANSCRIPT_URL = "https://api.openai.com/v1/audio/transcriptions"

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Supabase credentials missing")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- Helpers ----------------

def download_file(url: str) -> tuple[bytes, str]:
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "").lower()

def is_image_content(content_type: str) -> bool:
    return content_type.startswith("image/")

def extract_audio(video_bytes: bytes) -> bytes | None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as v:
        v.write(video_bytes)
        video_path = v.name

    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_path, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        if result.returncode != 0 or os.path.getsize(audio_path) < 1000:
            return None

        with open(audio_path, "rb") as f:
            return f.read()
    finally:
        os.remove(video_path)
        os.remove(audio_path)

def transcribe(video_bytes: bytes) -> dict | None:
    audio = extract_audio(video_bytes)
    if not audio:
        return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": ("audio.mp3", audio, "audio/mpeg")}
    data = {"model": "whisper-1", "response_format": "verbose_json"}

    r = requests.post(OPENAI_TRANSCRIPT_URL, headers=headers, files=files, data=data)
    if r.status_code != 200:
        return None
    return r.json()

def get_next_upload_number() -> int:
    r = supabase.table("transcript").select("uploadNumber").order(
        "uploadNumber", desc=True
    ).limit(1).execute()
    return (r.data[0]["uploadNumber"] + 1) if r.data else 1

# ---------------- Frame Storage ----------------

def save_frame(frame_bytes: bytes, transcript_id: int, timestamp: int):
    path = f"{transcript_id}/frame_{timestamp}.jpg"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(frame_bytes)
        tmp_path = tmp.name

    try:
        supabase.storage.from_("Ericc_video_frames").upload(
            path, tmp_path, {"content-type": "image/jpeg"}
        )

        url = supabase.storage.from_("Ericc_video_frames").get_public_url(path)
        if isinstance(url, dict):
            url = url.get("publicUrl")

        supabase.table("frame").insert({
            "transcript_id": transcript_id,
            "frame_timestamp": timestamp,
            "frame_storage_url": url
        }).execute()

        print(f"[OK] Frame saved → transcript {transcript_id}")

    except Exception as e:
        print("[FRAME ERROR]", e)

    finally:
        os.remove(tmp_path)

def process_video_frames(video_bytes: bytes, transcript_id: int):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as v:
        v.write(video_bytes)
        path = v.name

    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Video open failed")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = max(1, int(frames / fps))

        saved = False
        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                save_frame(buf.tobytes(), transcript_id, sec)
                saved = True

        if not saved:
            ret, frame = cap.read()
            if ret:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    save_frame(buf.tobytes(), transcript_id, 0)

        cap.release()
    finally:
        os.remove(path)

# ---------------- Upload API ----------------

@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json(force=True)

        name = data.get("full_name") or data.get("first_name")
        phone = data.get("phone") or data.get("Phone Number")
        files = data.get("File Upload") or data.get("Video Upload")

        if not name or not phone or not files:
            return jsonify({"error": "Missing fields"}), 400

        if isinstance(files, str):
            files = [files]

        upload_number = get_next_upload_number()
        transcript_ids = []

        for url in files:
            file_bytes, content_type = download_file(url)
            is_image = is_image_content(content_type)

            transcript = "[Image uploaded]" if is_image else "[No audio found]"

            resp = supabase.table("transcript").insert({
                "name": name,
                "phoneNumber": phone,
                "transcript": transcript,
                "uploadNumber": upload_number
            }).execute()

            transcript_id = resp.data[0]["id"]
            transcript_ids.append(transcript_id)

            if is_image:
                save_frame(file_bytes, transcript_id, 0)
            else:
                try:
                    result = transcribe(file_bytes)
                    if result:
                        text = "\n".join(
                            s["text"].strip() for s in result.get("segments", [])
                        )
                        supabase.table("transcript").update(
                            {"transcript": text}
                        ).eq("id", transcript_id).execute()
                except Exception:
                    pass

                threading.Thread(
                    target=process_video_frames,
                    args=(file_bytes, transcript_id),
                    daemon=True
                ).start()

        return jsonify({
            "success": True,
            "uploadNumber": upload_number,
            "transcript_ids": transcript_ids
        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Server error"}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
