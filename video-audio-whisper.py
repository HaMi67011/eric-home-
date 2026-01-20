from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
import os
import tempfile
import requests
import traceback
import cv2
import threading
import subprocess
from supabase import create_client, Client

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

# ---------------- Supabase ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- OpenAI ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPT_URL = "https://api.openai.com/v1/audio/transcriptions"

# ---------------- Download Video ----------------
def download_video(video_url: str) -> bytes:
    if not video_url.startswith("http"):
        raise ValueError("Invalid video URL")

    r = requests.get(video_url, timeout=60)
    r.raise_for_status()

    if not r.content:
        raise ValueError("Downloaded video is empty")

    return r.content

# ---------------- Timestamp ----------------
def format_timestamp(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{seconds//3600:02}:{(seconds//60)%60:02}:{seconds%60:02},{ms:03}"

# ---------------- Extract Audio ----------------
def extract_audio_from_video(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as v:
        v.write(video_bytes)
        video_path = v.name

    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", audio_path, "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )

        if result.returncode != 0:
            return None

        if os.path.getsize(audio_path) > 1000:
            with open(audio_path, "rb") as f:
                return f.read()

        return None
    finally:
        os.remove(video_path)
        os.remove(audio_path)

# ---------------- Transcription ----------------
def transcribe_video_verbose(video_bytes):
    audio_bytes = extract_audio_from_video(video_bytes)
    if not audio_bytes:
        return None

    files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg")}
    data = {"model": "whisper-1", "response_format": "verbose_json"}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    r = requests.post(OPENAI_TRANSCRIPT_URL, headers=headers, files=files, data=data)
    if r.status_code != 200:
        raise RuntimeError(r.text)

    return r.json()

def segments_to_text(segments):
    return "\n".join(
        f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text'].strip()}"
        for s in segments
    )

# ---------------- Frame Extraction (1/sec) ----------------
def process_frames_and_upload(video_bytes, transcript_id):
    frames = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as v:
        v.write(video_bytes)
        path = v.name

    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total / fps)

        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            file_path = f"{transcript_id}/frame_{sec}.jpg"

            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    file_path, buf.tobytes(), {"content-type": "image/jpeg"}
                )
                url = supabase.storage.from_("Ericc_video_frames").get_public_url(file_path)
            except Exception:
                url = None

            frames.append({
                "transcript_id": transcript_id,
                "frame_timestamp": sec,
                "frame_storage_url": url
            })

        cap.release()
    finally:
        os.remove(path)

    return frames

# ---------------- Background Worker ----------------
def background_frame_processing(video_bytes, transcript_id):
    try:
        frames = process_frames_and_upload(video_bytes, transcript_id)
        if frames:
            supabase.table("frame").insert(frames).execute()
    except Exception:
        print(traceback.format_exc())

# ---------------- Upload API ----------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json(force=True)

        name = data.get("full_name") or data.get("first_name")
        phone = data.get("phone") or data.get("Phone Number")
        video_url = data.get("customData", {}).get("video") or data.get("uploadvideolink")

        if not name or not phone or not video_url:
            return jsonify({"error": "Name, phone, and video are required"}), 400

        video_bytes = download_video(video_url)

        transcript_text = "[No audio found]"
        json_resp = transcribe_video_verbose(video_bytes)
        if json_resp:
            transcript_text = segments_to_text(json_resp.get("segments", []))

        resp = supabase.table("transcript").insert({
            "name": name,
            "phoneNumber": phone,
            "transcript": transcript_text
        }).execute()

        if not resp.data:
            return jsonify({"error": "Transcript save failed"}), 500

        transcript_id = resp.data[0]["id"]

        threading.Thread(
            target=background_frame_processing,
            args=(video_bytes, transcript_id),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "transcript_id": transcript_id,
            "frame_status": "processing",
#            "transcript": transcript_text
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

# ---------------- Errors ----------------
@app.errorhandler(RequestEntityTooLarge)
def large_file(e):
    return jsonify({"error": "File too large (15MB max)"}), 413

if __name__ == "__main__":
    app.run(debug=True)
