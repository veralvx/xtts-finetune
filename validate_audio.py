import json
import subprocess
from pathlib import Path


def run_ffprobe(file_path):
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error processing {file_path}: {e}")
        return None


def validate_audio(file_path, xtts_specs):
    data = run_ffprobe(file_path)
    if not data:
        return False, "FFprobe failed"

    audio_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "audio"), None
    )
    if not audio_stream:
        return False, "No audio stream found"

    sample_rate = int(audio_stream.get("sample_rate", 0))
    channels = int(audio_stream.get("channels", 0))
    duration = (
        float(audio_stream.get("duration", 0)) if audio_stream.get("duration") else 0
    )
    codec = audio_stream.get("codec_name", "")

    issues = []
    if sample_rate != xtts_specs["sample_rate"]:
        issues.append(
            f"Sample rate mismatch: {sample_rate} Hz (expected {xtts_specs['sample_rate']} Hz)"
        )
    if channels != xtts_specs["channels"]:
        issues.append(
            f"Channels mismatch: {channels} (expected {xtts_specs['channels']})"
        )
    if not xtts_specs["min_duration"] <= duration <= xtts_specs["max_duration"]:
        issues.append(
            f"Duration mismatch: {duration:.2f}s (expected {xtts_specs['min_duration']}-{xtts_specs['max_duration']}s)"
        )
    if not codec.startswith("pcm_s16le"):  # 16-bit PCM for WAV
        issues.append(f"Codec not ideal: {codec} (preferred pcm_s16le)")

    return len(issues) == 0, "; ".join(issues)


def main(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} not found.")
        return

    xtts_specs = {
        "sample_rate": 22050,
        "channels": 1,
        "min_duration": 1.0,
        "max_duration": 12.0,
    }

    audio_files = [
        f
        for f in folder.iterdir()
        if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ]
    if not audio_files:
        print("No supported audio files found.")
        return

    valid_count = 0
    for file in audio_files:
        is_valid, message = validate_audio(file, xtts_specs)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"{status} {file.name}: {message or 'All checks passed'}")
        if is_valid:
            valid_count += 1

    print(
        f"\nSummary: {valid_count}/{len(audio_files)} files compliant with XTTS v2 requirements."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python validate_audio.py <folder_path>")
    else:
        main(sys.argv[1])
