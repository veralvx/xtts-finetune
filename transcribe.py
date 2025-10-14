import argparse
import pathlib

import whisper

ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main(
    folder: pathlib.Path = pathlib.Path("dataset/wavs"),
    lang: str = "en",
    model: str = "medium",
    speaker: str = "speaker",
    device: str = "cuda",
    output: pathlib.Path | None = None,
) -> pathlib.Path:
    whisper_model = whisper.load_model(model, device=device)

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    audio_files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES
    ]
    audio_files = sorted(audio_files, key=lambda p: p.name)

    if not audio_files:
        print("No audio files found in the folder.")
        raise SystemExit(1)

    metadata_lines = []
    for audio_path in audio_files:
        result = whisper_model.transcribe(str(audio_path), language=lang, fp16=False)
        text = result.get("text", "").strip()
        file_id = audio_path.stem
        metadata_lines.append(f"{file_id}|{text}|{speaker}")

    if output is None:
        metadata_path = folder / "metadata.csv"
    else:
        out_path = output
        if out_path.suffix:
            metadata_path = out_path
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            metadata_path = out_path / "metadata.csv"

    metadata_path.write_text("\n".join(metadata_lines), encoding="utf-8")
    print(f"Transcription complete. Metadata saved to {metadata_path}")

    return metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe a folder of audio files to metadata.csv using Whisper."
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=pathlib.Path,
        default=pathlib.Path("dataset/wavs"),
        help="Folder with audio files",
    )
    parser.add_argument(
        "--lang", "-l", default="en", help="Language code for transcription (e.g. 'en')"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="medium",
        help="Whisper model name (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--speaker",
        "-s",
        default="speaker",
        help="Speaker label to add to metadata lines",
    )
    parser.add_argument(
        "--device", "-d", default="cuda", help="Device for whisper model (cuda or cpu)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=None,
        help="Output path. If a file (e.g. /tmp/out.csv) it's used directly; otherwise treated as a directory and metadata.csv is written inside.",
    )

    args = parser.parse_args()
    main(
        folder=args.folder,
        lang=args.lang,
        model=args.model,
        speaker=args.speaker,
        device=args.device,
        output=args.output,
    )
