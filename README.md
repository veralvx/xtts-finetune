# XTTS-FINETUNE 🚂

First, clone this repository and build the image:

```console
git clone https://github.com/veralvx/xtts-finetune xtts-finetune \
  && cd xtts-finetune \
  && podman build -f Dockerfile -t xtts-finetune
```

Start the container:

```console
podman run -it --rm --gpus=all -v ./dataset:/xtts/dataset xtts-finetune
```

Before fine-tuning, your directory layout should look like this:

```console
.
├── .venv
├── convert_audio.py
├── dataset
│   ├── metadata.csv
│   └── wavs
│       ├── 01.wav
│       ├── 02.wav
│       ├── reference.wav
│       └─ ...
├── Dockerfile
├── main.py
├── pyproject.toml
├── requirements.txt
├── finetune.py
├── transcribe.py
├── uv.lock
└── validate_audio.py
```

Notice:
- `.wav` files under `dataset/wavs`, with one file called `reference.wav` (~ 5s duration);
- `metadata.csv` under `dataset`

The `metadata.csv` can be obtained with:
 
```console
uv run main.py --transcribe ./dataset/wavs --lang en --model medium --device cuda
``` 

The metadata output will be under `dataset/wavs`, and it should be moved to `dataset/metadata.csv`.

Then:

```console
uv run main.py --lang en
```