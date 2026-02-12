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

def format_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    s = total_ms // 1000
    return f"{s//3600:02}:{(s//60)%60:02}:{s%60:02},{ms:03}"


def download_file(url: str) -> tuple[bytes, str]:
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "").lower()

def is_image_content(content_type: str) -> bool:
    return content_type.startswith("image/")

def extract_audio_from_video(video_bytes):
    """Extract audio from video file using ffmpeg. Returns audio bytes or None if no audio."""
    # Save video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_tmp:
        video_tmp.write(video_bytes)
        video_tmp.flush()
        video_path = video_tmp.name
    
    # Create temp file for audio output
    audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(audio_fd)
    
    try:
        # Use ffmpeg to extract audio
        # -i: input file
        # -vn: no video
        # -acodec libmp3lame: use mp3 codec
        # -ar 16000: sample rate 16kHz (good for Whisper)
        # -ac 1: mono audio
        # -b:a 64k: bitrate
        print(f"Running ffmpeg: {video_path} -> {audio_path}")
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "libmp3lame",
                "-ar", "16000", "-ac", "1", "-b:a", "64k",
                audio_path, "-y"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        # Log ffmpeg output for debugging
        if result.returncode != 0:
            print(f"ffmpeg error (return code {result.returncode}): {result.stderr.decode('utf-8', errors='ignore')[:500]}")
        
        # Check if audio file was created and has content
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"Audio file created: {file_size} bytes")
            
            if file_size > 1000:  # At least 1KB
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                return audio_bytes
            else:
                print("Audio file too small, likely no audio in video")
                return None
        else:
            print("Audio file was not created")
            return None
            
    except Exception as e:
        print(f"Audio extraction error: {str(e)}")
        print(traceback.format_exc())
        return None
    finally:
        # Cleanup temp files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

def transcribe(video_bytes: bytes) -> dict | None:
    audio = extract_audio_from_video(video_bytes)
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

        # ---------------- Extract fields ----------------
        name = data.get("full_name") or data.get("first_name")
        phone = data.get("phone") or data.get("Phone Number")
        files = data.get("File Upload") or data.get("Video Upload")
        full_address = data.get("full_address")
        city = data.get("city")
        state = data.get("state")
        country = data.get("country")

        if not name or not phone or not files:
            return jsonify({"error": "Missing fields"}), 400

        if isinstance(files, str):
            files = [files]

        upload_number = get_next_upload_number()
        transcript_ids = []

        # ---------------- Process each file ----------------
        for url in files:
            # Download file bytes
            file_bytes, content_type = download_file(url)
            is_image = is_image_content(content_type)

            # Default transcript
            transcript_text = "[Image uploaded]" if is_image else "[No audio found]"

            # ---------------- Transcribe if video ----------------
            if not is_image:
                try:
                    result = transcribe(file_bytes)
                    if result and result.get("segments"):
                        transcript_text = "\n".join(
                            f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text'].strip()}"
                            for s in result["segments"]
                        )
                except Exception as e:
                    print("Transcription error:", e)
                    print(traceback.format_exc())
                    # fallback transcript_text remains

            # ---------------- Insert transcript into Supabase ----------------
            resp = supabase.table("transcript").insert({
                "name": name,
                "phoneNumber": phone,
                "transcript": transcript_text,
                "uploadNumber": upload_number,
                "address": full_address,
                "city": city,
                "state": state,
                "country": country
            }).execute()

            transcript_id = resp.data[0]["id"]
            transcript_ids.append(transcript_id)

            # ---------------- Save frames ----------------
            if is_image:
                save_frame(file_bytes, transcript_id, 0)
            else:
                threading.Thread(
                    target=process_video_frames,
                    args=(file_bytes, transcript_id),
                    daemon=True
                ).start()

        # ---------------- Return response ----------------
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
