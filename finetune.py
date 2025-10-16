import gc
import os

import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig,
)

# torch.set_num_threads(os.cpu_count())

LOCAL_MODEL_DIR = (
    "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2"
)

DVAE_CHECKPOINT = os.path.join(LOCAL_MODEL_DIR, "dvae.pth")
MEL_NORM_FILE = os.path.join(LOCAL_MODEL_DIR, "mel_stats.pth")
TOKENIZER_FILE = os.path.join(LOCAL_MODEL_DIR, "vocab.json")
XTTS_CHECKPOINT = os.path.join(LOCAL_MODEL_DIR, "model.pth")

RUN_NAME = "GPT_XTTS_v2.0.2_LJSpeech_FT"  # Updated for v2.0.2
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"


LOGGER_URI = None

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 3  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
DATASETS_CONFIG_LIST, SPEAKER_REFERENCE, LANGUAGE = None, None, None


def main(device="gpu", mode=None, lang="en"):
    global DATASETS_CONFIG_LIST, SPEAKER_REFERENCE, LANGUAGE
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="mydataset",
        path="/xtts/dataset/",
        meta_file_train="metadata.csv",
        language=lang,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Training sentences generations
    SPEAKER_REFERENCE = [
        "/xtts/dataset/wavs/reference.wav"  # speaker reference to be used in training test sentences
    ]
    LANGUAGE = config_dataset.language

    plotstp = 100
    mxwavlen = 255995  # ~11.6 seconds
    txtln = 200
    batchgrpsize = 48
    evalmax = 256
    if device == "cpu" or mode == "lowvram":
        global BATCH_SIZE, GRAD_ACUMM_STEPS
        BATCH_SIZE = 1
        GRAD_ACUMM_STEPS = 252  # Adjust to maintain effective batch size ~252
        plotstp = 500
        # mxwavlen = 132300
        txtln = 150
        batchgrpsize = 1
        evalmax = 128

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=mxwavlen,
        max_text_length=txtln,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )
    # training parameters config
    config = GPTTrainerConfig(
        epochs=300,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="XTTS-v2.0.2-train",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=batchgrpsize,
        eval_batch_size=BATCH_SIZE,
        # mixed_precision=True if (device == "cpu" or mode == "lowvram") else False,
        mixed_precision=True,
        precision="bf16" if (device == "cpu") else "fp16",
        num_loader_workers=0 if (device == "cpu" and mode == "lowvram") else 8,
        eval_split_max_size=evalmax,
        print_step=50,
        plot_step=plotstp,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    full_samples, _ = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=False,
    )

    total_samples = len(full_samples)
    if total_samples <= 100:
        eval_split_size = (1 + 1e-9) / total_samples
    else:
        eval_split_size = 0.01

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        # eval_split_size=config.eval_split_size,  # DEFAULT
        eval_split_size=eval_split_size,
    )

    if mode == "lowvram":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gc.collect()

    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=START_WITH_EVAL,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    )

    # NEW: Configure for CPU training if needed
    if device == "cpu":
        print("Configuring for CPU training with bfloat16 mixed precision.")
        trainer_args.use_cpu = True
        trainer_args.bf16 = True
        trainer_args.mixed_precision = "bf16"
    elif mode == "lowvram" and device != "cpu":
        trainer_args.use_cpu = False
        trainer_args.mixed_precision = "fp16"

    trainer = Trainer(
        trainer_args,  # Pass the configured args object
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

    del model, trainer, train_samples, eval_samples
    gc.collect()


if __name__ == "__main__":
    main()
