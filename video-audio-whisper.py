from flask import Flask, request, jsonify, render_template, Response
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
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max upload

# ---------------- Supabase Connection ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- OpenAI Settings ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPT_URL = "https://api.openai.com/v1/audio/transcriptions"

# ---------------- Helper: Extract Audio from Video ----------------
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
def transcribe_video_verbose(video_bytes):
    """
    Extract audio from video and call OpenAI Whisper API with verbose_json.
    Returns None if video has no audio.
    """
    if not video_bytes:
        raise ValueError("Uploaded video file is empty")
    
    # Extract audio from video
    print("Extracting audio from video...")
    audio_bytes = extract_audio_from_video(video_bytes)
    
    if not audio_bytes:
        print("No audio found in video")
        return None
    
    print(f"Audio extracted: {len(audio_bytes)} bytes")

    # Send audio to OpenAI Whisper API
    # Important: Use proper multipart/form-data format
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
    
    print(f"Frame extraction: Saving video to temp file...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    print(f"Frame extraction: Temp file created at {tmp_path}")

    try:
        print(f"Frame extraction: Opening video with OpenCV...")
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)
        
        print(f"Frame extraction: Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {duration}s")

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
            except Exception as upload_error:
                print(f"Frame upload error for {file_path}: {str(upload_error)}")
                public_url = None

            frames_metadata.append({
                "transcript_id": transcript_id,
                "frame_timestamp": round(frame_ts, 3),
                "frame_storage_url": public_url
            })

        cap.release()
        print(f"Frame extraction: Successfully extracted {len(frames_metadata)} frames")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Frame extraction: Cleaned up temp file")

    return frames_metadata

# ---------------- Background Task for Frame Processing ----------------
def background_frame_processing(file_bytes, transcript_id):
    """Process frames in the background to avoid blocking the request."""
    print(f"Background: Starting frame processing for transcript {transcript_id}")
    print(f"Background: Video bytes size: {len(file_bytes)}")
    
    try:
        frames = process_frames_and_upload(file_bytes, transcript_id)
        print(f"Background: Frame extraction completed. Found {len(frames)} frames")
        
        if frames:
            print(f"Background: Inserting {len(frames)} frames into database...")
            supabase.table("frame").insert(frames).execute()
            print(f"Background: Successfully saved {len(frames)} frames for transcript {transcript_id}")
        else:
            print(f"Background: No frames extracted for transcript {transcript_id}")
            
    except Exception as e:
        print(f"Background frame processing error for transcript {transcript_id}: {str(e)}")
        print(f"Background error traceback:\n{traceback.format_exc()}")


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
        timestamped_transcript = ""
        try:
            json_resp = transcribe_video_verbose(file_bytes)
            
            # Check if video has audio
            if json_resp is None:
                timestamped_transcript = "[No audio found in video]"
            else:
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

        # ---------------- Process frames in background ----------------
        # Start frame processing in a background thread to avoid timeout
        thread = threading.Thread(
            target=background_frame_processing,
            args=(file_bytes, transcript_id)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "message": "Video processed successfully. Frames are being extracted in the background.",
            "transcript_id": transcript_id,
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
