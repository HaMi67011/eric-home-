from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
import os
import tempfile
import requests
import traceback
import subprocess
import glob

from supabase import create_client, Client

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max upload

# ---------------- Supabase ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- OpenAI ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPT_URL = "https://api.openai.com/v1/audio/transcriptions"

# ---------------- Timestamp helper ----------------
def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    return f"{h:02}:{m:02}:{s:02},{millis:03}"

# ---------------- Whisper ----------------
def transcribe_video_verbose(file_bytes):
    files = {"file": ("video.mp4", file_bytes)}
    data = {"model": "whisper-1", "response_format": "verbose_json"}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    r = requests.post(OPENAI_TRANSCRIPT_URL, headers=headers, files=files, data=data)
    if r.status_code != 200:
        raise ValueError(r.text)

    return r.json()

# ---------------- Transcript formatter ----------------
def segments_to_timestamped_text(segments):
    out = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        out.append(f"[{start} --> {end}] {seg['text'].strip()}")
    return "\n".join(out)

# ---------------- FFmpeg Frame Extraction ----------------
def process_frames_and_upload(file_bytes, transcript_id):
    frames = []

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        with open(video_path, "wb") as f:
            f.write(file_bytes)

        # Extract 1 frame per second
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", "fps=1",
            os.path.join(frames_dir, "frame_%04d.jpg")
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        images = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

        for i, img_path in enumerate(images):
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            file_path = f"{transcript_id}/frame_{i}.jpg"

            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    file_path, img_bytes, {"content-type": "image/jpeg"}
                )
                public_url = supabase.storage.from_("Ericc_video_frames").get_public_url(file_path)
            except:
                public_url = None

            frames.append({
                "transcript_id": transcript_id,
                "frame_timestamp": i,
                "frame_storage_url": public_url
            })

    return frames

# ---------------- Upload ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        name = request.form.get("name")
        phone = request.form.get("phone")
        file = request.files.get("video")

        if not name or not phone or not file:
            return jsonify({"error": "Missing data"}), 400

        file_bytes = file.read()

        # Transcribe
        whisper_json = transcribe_video_verbose(file_bytes)
        segments = whisper_json.get("segments", [])
        transcript = segments_to_timestamped_text(segments)

        # Store transcript
        resp = supabase.table("transcript").insert({
            "name": name,
            "phoneNumber": phone,
            "transcript": transcript
        }).execute()

        transcript_id = resp.data[0]["id"]

        # Extract frames
        frames = process_frames_and_upload(file_bytes, transcript_id)

        if frames:
            supabase.table("frame").insert(frames).execute()

        return jsonify({
            "message": "Success",
            "transcript_id": transcript_id,
            "frame_count": len(frames),
            "transcript": transcript
        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Processing failed"}), 500

# ---------------- Large files ----------------
@app.errorhandler(RequestEntityTooLarge)
def too_big(e):
    return jsonify({"error": "File too large"}), 413

# ---------------- Web ----------------
@app.route("/")
def home():
    return render_template("main.html")

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
