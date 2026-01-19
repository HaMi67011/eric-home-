from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
import os
import tempfile
import requests
import traceback
import cv2

from supabase import create_client, Client

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max upload

# ---------------- Supabase Connection ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- OpenAI Settings ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPT_URL = "https://api.openai.com/v1/audio/transcriptions"

# ---------------- Helper: Format Timestamp ----------------
def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    return f"{h:02}:{m:02}:{s:02},{millis:03}"

# ---------------- Download Video From URL ----------------
def download_video_from_url(video_url):
    # If video_url is a list (from GHL), take first element
    if isinstance(video_url, list):
        video_url = video_url[0]

    if not video_url.startswith("http"):
        raise ValueError("Invalid video URL")

    resp = requests.get(video_url, stream=True, timeout=30)
    resp.raise_for_status()
    return resp.content

# ---------------- Transcribe Video ----------------
def transcribe_video_verbose(file_bytes):
    if not file_bytes:
        raise ValueError("Video is empty")

    files = {"file": ("video.mp4", file_bytes)}
    data = {"model": "whisper-1", "response_format": "verbose_json"}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    response = requests.post(
        OPENAI_TRANSCRIPT_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=120
    )

    response.raise_for_status()
    return response.json()

# ---------------- Convert Segments ----------------
def segments_to_timestamped_text(segments):
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start} --> {end}] {text}")
    return "\n".join(lines)

# ---------------- Process Frames ----------------
def process_frames_and_upload(file_bytes, transcript_id):
    frames_metadata = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        video_path = tmp.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            ts = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 3)
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            image_bytes = buffer.tobytes()
            path = f"{transcript_id}/frame_{sec}.jpg"

            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    path,
                    image_bytes,
                    {"content-type": "image/jpeg"}
                )
                url = supabase.storage.from_("Ericc_video_frames").get_public_url(path)
            except Exception:
                url = None

            frames_metadata.append({
                "transcript_id": transcript_id,
                "frame_timestamp": ts,
                "frame_storage_url": url
            })

        cap.release()
    finally:
        os.remove(video_path)

    return frames_metadata

# ---------------- Upload Route (API Only) ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # Accept JSON or form-data
        data = request.get_json() if request.is_json else request.form

        name = data.get("full_name") or data.get("first_name")
        phone = data.get("phone")
        video_url = data.get("video")
        file = request.files.get("video")

        if not name or not phone:
            return jsonify({"error": "Name and phone are required"}), 400

        # --- Get video bytes ---
        if file:
            file_bytes = file.read()
        elif video_url:
            file_bytes = download_video_from_url(video_url)
        else:
            return jsonify({"error": "No video or video_url provided"}), 400

        if not file_bytes:
            return jsonify({"error": "Video is empty"}), 400

        # --- Transcription ---
        json_resp = transcribe_video_verbose(file_bytes)
        segments = json_resp.get("segments", [])
        transcript_text = segments_to_timestamped_text(segments)

        # --- Save transcript ---
        transcript_payload = {
            "name": name,
            "phoneNumber": str(phone),
            "transcript": transcript_text
        }

        transcript_resp = supabase.table("transcript").insert(transcript_payload).execute()
        transcript_id = transcript_resp.data[0]["id"]

        # --- Process frames ---
        frames = process_frames_and_upload(file_bytes, transcript_id)
        if frames:
            supabase.table("frame").insert(frames).execute()

        return jsonify({
            "message": "Video processed successfully",
            "transcript_id": transcript_id,
            "frame_count": len(frames),
            "transcript": transcript_text
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ---------------- Error: File Too Large ----------------
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File too large. Max 15MB."}), 413

# ---------------- Run API ----------------
if __name__ == "__main__":
    app.run(debug=True)
