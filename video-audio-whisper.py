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
    raise RuntimeError("Supabase credentials not found")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- Helpers ----------------

def download_video(video_url: str) -> bytes:
    if not video_url.startswith("http"):
        raise ValueError("Invalid video URL")

    r = requests.get(video_url, timeout=60)
    r.raise_for_status()

    if not r.content:
        raise ValueError("Downloaded video is empty")

    return r.content


def format_timestamp(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{seconds//3600:02}:{(seconds//60)%60:02}:{seconds%60:02},{ms:03}"


def extract_audio_from_video(video_bytes: bytes) -> bytes | None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as v:
        v.write(video_bytes)
        video_path = v.name

    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-b:a", "64k", audio_path, "-y"
            ],
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


def transcribe_video_verbose(video_bytes: bytes) -> dict | None:
    audio_bytes = extract_audio_from_video(video_bytes)
    if not audio_bytes:
        return None

    files = {
        "file": ("audio.mp3", audio_bytes, "audio/mpeg")
    }

    data = {
        "model": "whisper-1",
        "response_format": "verbose_json"
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    r = requests.post(
        OPENAI_TRANSCRIPT_URL,
        headers=headers,
        files=files,
        data=data
    )

    if r.status_code != 200:
        raise RuntimeError(r.text)

    return r.json()


def segments_to_text(segments: list) -> str:
    return "\n".join(
        f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text'].strip()}"
        for s in segments
    )


def get_next_upload_number() -> int:
    resp = (
        supabase
        .table("transcript")
        .select("uploadNumber")
        .is_("uploadNumber", "not_null")  # ✅ underscore, not dot
        .order("uploadNumber", desc=True)
        .limit(1)
        .execute()
    )

    if resp.data:
        return int(resp.data[0]["uploadNumber"]) + 1

    return 1



# ---------------- Frame Processing ----------------

def process_frames_and_upload(video_bytes: bytes, transcript_id: int) -> list:
    frames = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as v:
        v.write(video_bytes)
        video_path = v.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            storage_path = f"{transcript_id}/frame_{sec}.jpg"

            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    storage_path,
                    buf.tobytes(),
                    {"content-type": "image/jpeg"}
                )

                url = supabase.storage.from_("Ericc_video_frames").get_public_url(storage_path)
            except Exception:
                url = None

            frames.append({
                "transcript_id": transcript_id,
                "frame_timestamp": sec,
                "frame_storage_url": url
            })

        cap.release()

    finally:
        os.remove(video_path)

    return frames


def background_frame_processing(video_bytes: bytes, transcript_id: int):
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
        video_urls = data.get("Video Upload")

        if not video_urls:
            video_urls = []
        elif isinstance(video_urls, str):
            video_urls = [video_urls]

        if not name or not phone or not video_urls:
            return jsonify({"error": "Name, phone, and at least one video are required"}), 400

        # 🔹 One global uploadNumber for this request
        upload_number = get_next_upload_number()

        transcript_ids = []

        for video_url in video_urls:
            try:
                video_bytes = download_video(video_url)

                transcript_text = "[No audio found]"
                json_resp = transcribe_video_verbose(video_bytes)
                if json_resp:
                    transcript_text = segments_to_text(json_resp.get("segments", []))

                resp = supabase.table("transcript").insert({
                    "name": name,
                    "phoneNumber": phone,
                    "transcript": transcript_text,
                    "uploadNumber": upload_number
                }).execute()

                if resp.data:
                    transcript_id = resp.data[0]["id"]
                    transcript_ids.append(transcript_id)

                    threading.Thread(
                        target=background_frame_processing,
                        args=(video_bytes, transcript_id),
                        daemon=True
                    ).start()

            except Exception as e:
                print(f"Failed to process video: {video_url}\n{e}")

        return jsonify({
            "success": True,
            "uploadNumber": upload_number,
            "transcripts_processed": len(transcript_ids),
            "transcript_ids": transcript_ids,
            "frame_status": "processing"
        }), 200

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

# ---------------- Run App ----------------

if __name__ == "__main__":
    app.run(debug=True)
