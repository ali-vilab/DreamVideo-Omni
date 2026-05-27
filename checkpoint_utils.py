import os
from collections import OrderedDict

import torch
from safetensors.torch import load_file as load_safetensors


def load_checkpoint_to_pipeline(pipe, checkpoint_path: str, device="cpu"):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_safetensors(checkpoint_path, device=device)
    else:
        loaded = torch.load(checkpoint_path, map_location=device)
        state_dict = loaded["module"] if isinstance(loaded, dict) and "module" in loaded else loaded

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("pipe.") and not key.startswith("pipe.dit."):
            continue
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("pipe.dit."):
            new_key = new_key.replace("pipe.dit.", "")
        elif new_key.startswith("dit."):
            new_key = new_key.replace("dit.", "")
        if "pipe.bbox_zeroconv" in key:
            if "bbox_zeroconv" in new_key and "dit" not in new_key:
                pass
            else:
                new_key = key.replace("pipe.bbox_zeroconv", "bbox_zeroconv")
        if new_key.startswith("pipe."):
            new_key = new_key.replace("pipe.", "")
        new_state_dict[new_key] = value

    missing, unexpected = pipe.dit.load_state_dict(new_state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing[:5]}...")
    print(f"[Info] Loaded checkpoint: {checkpoint_path}")
    if unexpected:
        print(f"[Info] Unexpected keys: {len(unexpected)}")
