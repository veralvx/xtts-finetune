import json
import subprocess
import tempfile
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
        str(file_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error processing {file_path}: {e}")
        return None


def convert_audio(file_path: Path, target_specs: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=file_path.parent, suffix=".wav")
    tmp_path = Path(tmp.name)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(file_path),
        "-ar",
        str(target_specs["sample_rate"]),
        "-ac",
        str(target_specs["channels"]),
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # cleanup temp file if conversion failed
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        print(f"Conversion failed for {file_path}: {e}\nstderr: {e.stderr}")
        return False

    # Replace original with the temp file atomically
    try:
        # os.replace is atomic on the same filesystem
        Path(str(tmp_path)).replace(str(file_path))
        print(f"Converted and replaced: {file_path}")
        return True
    except Exception as e:
        # If replace fails, remove the temp and report
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        print(f"Failed to replace original with converted file for {file_path}: {e}")
        return False


def check_and_convert(file_path, xtts_specs):
    data = run_ffprobe(file_path)
    if not data:
        return False

    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if not audio_stream:
        print(f"No audio stream found in {file_path}")
        return False

    sample_rate = int(audio_stream.get("sample_rate", 0))
    channels = int(audio_stream.get("channels", 0))
    codec = audio_stream.get("codec_name", "")

    needs_conversion = (
        sample_rate != xtts_specs["sample_rate"]
        or channels != xtts_specs["channels"]
    )

    if needs_conversion:
        return convert_audio(file_path, xtts_specs)
    print(f"Already compliant: {file_path}")
    return True


def main(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} not found.")
        return

    # XTTS v2 input specs (excluding duration)
    xtts_specs = {"sample_rate": 22050, "channels": 1}

    audio_files = [
        f
        for f in folder.iterdir()
        if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ]
    if not audio_files:
        print("No supported audio files found.")
        return

    success_count = 0
    for file in audio_files:
        if check_and_convert(file, xtts_specs):
            success_count += 1

    print(
        f"\nSummary: {success_count}/{len(audio_files)} files processed successfully."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python convert_audio.py <folder_path>")
    else:
        main(sys.argv[1])
