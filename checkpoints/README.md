# Weights

Place the DreamVideo-Omni DiT (SFT) checkpoint in this directory before running inference:

```text
checkpoints/dreamvideo_omni_sft.safetensors
```

## Download

**Hugging Face**

```bash
huggingface-cli download weilllllls/DreamVideo-Omni dreamvideo_omni_sft.safetensors \
  --local-dir ./checkpoints
```

**ModelScope**

```bash
modelscope download --model weilllllls/DreamVideo-Omni \
  dreamvideo_omni_sft.safetensors --local_dir ./checkpoints
```

Wan2.1-T2V-1.3B base weights are loaded from `--local_model_path` (default `./models`) or auto-downloaded from ModelScope on the first run. See the main [README.md](../README.md) for Wan base weight download commands.
