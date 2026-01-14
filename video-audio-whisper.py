from flask import Flask, request, jsonify, render_template, Response
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
    """Convert seconds (float) to H:M:S,ms format."""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    return f"{h:02}:{m:02}:{s:02},{millis:03}"

# ---------------- Transcribe Video via OpenAI HTTP ----------------
def transcribe_video_verbose(file_bytes):
    """Call OpenAI Whisper API with verbose_json to get segments with timestamps."""
    if not file_bytes:
        raise ValueError("Uploaded video/audio file is empty")

    files = {"file": ("video.mp4", file_bytes)}
    data = {"model": "whisper-1", "response_format": "verbose_json"}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    response = requests.post(OPENAI_TRANSCRIPT_URL, headers=headers, files=files, data=data)
    if response.status_code != 200:
        raise ValueError(f"Transcription failed: {response.text}")

    return response.json()

# ---------------- Convert segments to timestamped transcript ----------------
def segments_to_timestamped_text(segments):
    """Return a string where each segment has [start --> end] prefix."""
    lines = []
    for seg in segments:
        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start_ts} --> {end_ts}] {text}")
    return "\n".join(lines)

# ---------------- Process frames + upload to Supabase ----------------
def process_frames_and_upload(file_bytes, transcript_id):
    frames_metadata = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_ts = int(round(frame_ts))  # now an integer
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            image_bytes = buffer.tobytes()
            file_path = f"{transcript_id}/frame_{sec}.jpg"

            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    file_path, image_bytes, {"content-type": "image/jpeg"}
                )
                public_url = supabase.storage.from_("Ericc_video_frames").get_public_url(file_path)
            except Exception:
                public_url = None

            frames_metadata.append({
                "transcript_id": transcript_id,
                "frame_timestamp": round(frame_ts, 3),
                "frame_storage_url": public_url
            })

        cap.release()
    finally:
        os.remove(tmp_path)

    return frames_metadata

# ---------------- Upload Route ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        name = request.form.get("name")
        phone_str = request.form.get("phone")

        if not name or not phone_str:
            return jsonify({"error": "Name and phone are required"}), 400

        file = request.files.get("video")
        if not file:
            return jsonify({"error": "No video file provided"}), 400

        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Uploaded video is empty"}), 400

        # ---------------- Transcription ----------------
        try:
            json_resp = transcribe_video_verbose(file_bytes)
            segments = json_resp.get("segments", [])
            timestamped_transcript = segments_to_timestamped_text(segments)
        except Exception as e:
            print("Transcription error:\n", traceback.format_exc())
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        # ---------------- Insert transcript ----------------
        transcript_payload = {
            "name": name,
            "phoneNumber": str(phone_str),
            "transcript": timestamped_transcript
        }

        transcript_resp = supabase.table("transcript").insert(transcript_payload).execute()
        if not transcript_resp.data:
            return jsonify({"error": "Failed to save transcript"}), 500

        transcript_id = transcript_resp.data[0]["id"]

        # ---------------- Process frames ----------------
        frames = []
        try:
            frames = process_frames_and_upload(file_bytes, transcript_id)
        except Exception as e:
            print("Frame processing/upload error:\n", traceback.format_exc())

        # ---------------- Insert frame metadata ----------------
        if frames:
            try:
                supabase.table("frame").insert(frames).execute()
            except Exception:
                pass

        return jsonify({
            "message": "Video processed successfully",
            "transcript_id": transcript_id,
            "frame_count": len(frames),
            "transcript": timestamped_transcript
        })

    except Exception as e:
        print("Unexpected exception:\n", traceback.format_exc())
        return jsonify({"error": f"Upload or transcription failed: {str(e)}"}), 500

# ---------------- Handle Large Uploads ----------------
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File too large. Max 15 MB allowed."}), 413

# ---------------- Serve HTML ----------------
@app.route("/")
def index():
    return render_template("main.html")

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
