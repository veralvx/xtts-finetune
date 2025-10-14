#ARG BASE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG BASE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM ${BASE}
#ENV UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends bash coreutils vim nano gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel curl wget git git-lfs espeak-ng ffmpeg flac libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN git lfs install

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"

RUN rm -rf /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2
RUN git clone -b v2.0.2 https://huggingface.co/coqui/XTTS-v2 /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2

WORKDIR  /xtts
COPY pyproject.toml uv.lock ./
RUN uv python install 3.11 && uv python pin 3.11 && uv venv .venv --python=3.11
ENV PATH="/xtts/.venv/bin:$PATH"
# RUN uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
#RUN uv add -r requirements-uv.txt; 

RUN uv sync --compile-bytecode --frozen
RUN uv run --with openai-whisper -- python3 -c "import whisper;  whisper.load_model('large'); whisper.load_model('medium'); whisper.load_model('turbo'); whisper.load_model('small')"
#RUN git config user.name "You"  && git config user.email "you@example.com" && git add -A && git commit -m "initial commit"
COPY  . .

#ENTRYPOINT ["/bin/sh", "-c", ". .venv/bin/activate && sh"]
#ENTRYPOINT ["/bin/sh", "-c", ". .venv/bin/activate && exec \"$@\"" ]