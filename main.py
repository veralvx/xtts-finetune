import optparse

parser = optparse.OptionParser()
parser.add_option(
    "--device",
    dest="device",
    default="cuda",
    help="Device to use: cpu or cuda (for fine-tuning)",
)
parser.add_option(
    "--mode",
    dest="mode",
    default=None,
    help="None or lowvram",
)
parser.add_option(
    "--transcribe",
    dest="transcribe",
    default=None,
    help="Folder path to transcribe audio files",
)
parser.add_option(
    "--lang",
    dest="lang",
    default="en",
    help="Language for transcription (e.g., en, fr)",
)
parser.add_option(
    "--model",
    dest="model",
    default="medium",
    help="Whisper model size (e.g., tiny, base, small, medium, large)",
)
parser.add_option(
    "--speaker", dest="speaker", default="speaker", help="Speaker name for metadata"
)
parser.add_option(
    "--output", dest="output", default="dataset", help="Output Folder for metadata.csv"
)
parser.add_option(
    "--validate",
    dest="validate",
    default=None,
    help="Folder path to validate audio files",
)
parser.add_option(
    "--convert", dest="convert", default=None, help="Folder path to convert audio files"
)

(options, args) = parser.parse_args()

if options.transcribe:
    import transcribe

    transcribe.main(
        options.transcribe,
        options.lang,
        options.model,
        options.speaker,
        options.device,
        options.output,
    )
elif options.validate:
    import validate_audio

    validate_audio.main(options.validate)
elif options.convert:
    import convert_audio

    convert_audio.main(options.convert)
else:
    import finetune

    device = "gpu" if options.device != "cpu" else "cpu"
    finetune.main(device=device, mode=options.mode, lang=options.lang)
