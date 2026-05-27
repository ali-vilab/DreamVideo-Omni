# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.12257-b31b1b.svg)](https://arxiv.org/abs/2603.12257)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://dreamvideo-omni.github.io/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-DreamVideo--Omni-yellow)](https://huggingface.co/weilllllls/DreamVideo-Omni)
[![ModelScope](https://img.shields.io/badge/ModelScope-DreamVideo--Omni-blue)](https://modelscope.cn/models/weilllllls/DreamVideo-Omni)

**[Yujie Wei<sup>1</sup>](https://weilllllls.github.io), [Xinyu Liu<sup>2</sup>](https://scholar.google.com/citations?user=kgRjFN8AAAAJ), [Shiwei Zhang<sup>3</sup>](https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ), [Hangjie Yuan<sup>4</sup>](https://jacobyuan7.github.io), [Jinbo Xing<sup>3</sup>](https://doubiiu.github.io/), [Zhekai Chen<sup>5</sup>](https://scholar.google.com/citations?user=_eZWcIMAAAAJ), [Xiang Wang<sup>3</sup>](https://scholar.google.com/citations?user=cQbXvkcAAAAJ), [Haonan Qiu<sup>6</sup>](http://haonanqiu.com/), [Rui Zhao<sup>7</sup>](https://ruizhaocv.github.io/), [Yutong Feng<sup>3</sup>](https://scholar.google.com/citations?user=mZwJLeUAAAAJ), [Ruihang Chu<sup>3</sup>](https://ruihang-chu.github.io/), [Yingya Zhang<sup>3</sup>](https://scholar.google.com.sg/citations?user=16RDSEUAAAAJ), [Yike Guo<sup>2</sup>](https://cse.hkust.edu.hk/admin/people/faculty/profile/yikeguo), [Xihui Liu<sup>5</sup>](https://xh-liu.github.io/), [Hongming Shan<sup>1</sup>](http://hmshan.io)**

<sup>1</sup>Fudan University &nbsp; <sup>2</sup>The Hong Kong University of Science and Technology &nbsp; <sup>3</sup>Tongyi Lab, Alibaba Group <br> <sup>4</sup>Zhejiang University &nbsp; <sup>5</sup>MMLab, The University of Hong Kong <br> <sup>6</sup>Nanyang Technological University &nbsp; <sup>7</sup>National University of Singapore


</div>

## Abstract

While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control. In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm. In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements. To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances. In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pre-trained video diffusion backbone. This provides motion-aware identity rewards in the latent space, prioritizing identity preservation aligned with human preferences. Supported by our curated large-scale dataset and the comprehensive DreamOmni Bench for multi-subject and omni-motion control evaluation, DreamVideo-Omni demonstrates superior performance in generating high-quality videos with precise controllability. Project page: [https://dreamvideo-omni.github.io](https://dreamvideo-omni.github.io).


## 🔥 Updates

- __[2026.05]__: Release the inference code and SFT checkpoint.
- __[2026.03]__: Release the [paper](https://arxiv.org/abs/2603.12257) of DreamVideo-Omni.


## ⚙️ Preparation

### 1. Requirements & Installation

```bash
conda create -n dreamvideo-omni python=3.10 -y
conda activate dreamvideo-omni
pip install -e .
```

### 2. Download Weights

#### DreamVideo-Omni DiT (SFT) checkpoint

Download `dreamvideo_omni_sft.safetensors` (~2.8 GB) to:

```text
checkpoints/dreamvideo_omni_sft.safetensors
```

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

#### Wan2.1-T2V-1.3B base weights

Required for text encoder, VAE, and tokenizer. If not pre-downloaded, they will be fetched automatically from ModelScope on the first inference run.

**Hugging Face**

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir ./models/Wan-AI/Wan2.1-T2V-1.3B
```

**ModelScope**

```bash
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B \
  --local_dir ./models/Wan-AI/Wan2.1-T2V-1.3B
```

#### Use local weights without re-downloading

If you already have Wan base weights locally:

```bash
export LOCAL_MODEL_PATH=/path/to/pretrain_models_ckpt
export SKIP_DOWNLOAD=1
```

Then pass `--skip_download` to `infer.py`, or set `SKIP_DOWNLOAD=1` when running the shell scripts.

## 💫 Inference

Video generation is performed via `infer.py`. Each example case under `examples/` contains a `metadata.json` with caption, reference images, bounding boxes, and/or trajectories.

### Three example cases

| Case | Directory | Control type | Seed |
|------|-----------|--------------|------|
| **0** | `examples/0` | **Multi-reference** — two reference images, text prompt only (no tracks / bbox) | `555362` |
| **1** | `examples/1` | **Motion** — trajectory + per-frame `frames_info` bbox (no reference image) | `42` |
| **2** | `examples/2` | **Identity + motion** — one reference image + trajectory + per-frame bbox | `45` |


### Commands

**Case 0 — multi-reference**

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint ./checkpoints/dreamvideo_omni_sft.safetensors \
  --local_model_path ./models \
  --case_dir examples/0 \
  --output_path outputs/0.mp4 \
  --num_inference_steps 50 \
  --seed 555362 \
  --skip_download
```

**Case 1 — motion (tracks + bbox)**

```bash
CUDA_VISIBLE_DEVICES=1 python infer.py \
  --checkpoint ./checkpoints/dreamvideo_omni_sft.safetensors \
  --local_model_path ./models \
  --case_dir examples/1 \
  --output_path outputs/1.mp4 \
  --num_inference_steps 50 \
  --seed 42 \
  --skip_download
```

**Case 2 — identity + motion (ref + tracks + bbox)**

```bash
CUDA_VISIBLE_DEVICES=2 python infer.py \
  --checkpoint ./checkpoints/dreamvideo_omni_sft.safetensors \
  --local_model_path ./models \
  --case_dir examples/2 \
  --output_path outputs/2.mp4 \
  --num_inference_steps 50 \
  --seed 45 \
  --skip_download
```

**Notes:**

- Each case folder contains `metadata.json` (caption, paths to assets, optional `frames_info`).
- `frames_info` supplies per-frame bounding boxes; mask PNGs are optional via `paths.masks`.
- `ref_imgs[].obj_id` and bbox `obj_id` should use consistent sort order for multi-object setups.

## Acknowledgements

This code is built on top of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). We thank the authors for their great work.

## 🌟 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{wei2026dreamvideo_omni,
  title={DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning},
  author={Wei, Yujie and Liu, Xinyu and Zhang, Shiwei and Yuan, Hangjie and Xing, Jinbo and Chen, Zhekai and Wang, Xiang and Qiu, Haonan and Zhao, Rui and Feng, Yutong and Chu, Ruihang and Zhang, Yingya and Guo, Yike and Liu, Xihui and Shan, Hongming},
  journal={arXiv preprint arXiv:2603.12257},
  year={2026}
}
```
