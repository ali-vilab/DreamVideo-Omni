import glob
import os
import re

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

COLOR_PALETTE = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
]


def process_ref_image(image_path, target_width=832, target_height=480):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    return Image.open(image_path).convert("RGB").resize((target_width, target_height), Image.LANCZOS)


def _obj_id_sort_key(obj_id):
    oid = str(obj_id)
    try:
        return (0, int(oid))
    except ValueError:
        return (1, oid)


def ordered_obj_ids(data):
    ids = set()
    for frame in data.get("frames_info") or []:
        for box in frame.get("bboxes", []):
            ids.add(str(box["obj_id"]))
    for entry in data.get("paths", {}).get("ref_imgs") or []:
        if entry.get("obj_id") is not None:
            ids.add(str(entry["obj_id"]))
    return sorted(ids, key=_obj_id_sort_key)


def load_refs_from_metadata(case_folder, data, ordered_ids, target_width=832, target_height=480):
    ref_info = data.get("paths", {}).get("ref_imgs") or []
    if not ref_info:
        return [], []

    by_id = {}
    for entry in ref_info:
        oid = str(entry["obj_id"])
        by_id.setdefault(oid, []).append(entry)

    ref_images = []
    for oid in ordered_ids:
        entries = by_id.get(oid)
        if not entries:
            raise ValueError(
                f"metadata ref_imgs missing obj_id={oid!r}; "
                f"expected refs for ordered ids {ordered_ids}"
            )
        entry = sorted(entries, key=lambda e: e.get("frame_idx", 0))[0]
        ref_path = os.path.join(case_folder, entry["path"])
        ref_images.append(process_ref_image(ref_path, target_width, target_height))

    if not ref_images:
        return [], []
    return ref_images, [len(ref_images)]


def build_track_channel_obj_slot(data, num_tracks, num_obj_slots):
    obj_track_idx = data.get("obj_track_idx")
    if obj_track_idx is None:
        return None

    if len(obj_track_idx) != num_obj_slots:
        raise ValueError(
            f"obj_track_idx length {len(obj_track_idx)} != num object slots {num_obj_slots}"
        )

    mapping = [-1] * num_tracks
    for obj_slot, track_idx in enumerate(obj_track_idx):
        track_idx = int(track_idx)
        if track_idx < 0 or track_idx >= num_tracks:
            raise ValueError(
                f"obj_track_idx[{obj_slot}]={track_idx} out of range for {num_tracks} tracks"
            )
        if mapping[track_idx] >= 0:
            raise ValueError(f"track index {track_idx} bound to more than one object slot")
        mapping[track_idx] = int(obj_slot)
    return mapping


def extract_multi_ref_imgs(reference_images_dir, target_width=832, target_height=480):
    ref_pattern = os.path.join(reference_images_dir, "ref_*.png")
    all_ref_files = glob.glob(ref_pattern)
    if not all_ref_files:
        raise FileNotFoundError(f"No reference images in {reference_images_dir}")

    obj_dict = {}
    for file_path in all_ref_files:
        filename = os.path.basename(file_path)
        match = re.match(r"ref_(\d+)_(.+)_frame_(\d+)\.png", filename)
        if not match:
            continue
        obj_id = int(match.group(1))
        obj_name = match.group(2)
        frame_num = int(match.group(3))
        obj_dict.setdefault(obj_id, []).append((frame_num, file_path, obj_name))

    ref_images_list = []
    obj_info_list = []
    for obj_id in sorted(obj_dict.keys()):
        frames = sorted(obj_dict[obj_id], key=lambda x: x[0])
        _, first_frame_path, obj_name = frames[0]
        obj_info_list.append((obj_id, obj_name, first_frame_path))
        ref_images_list.append(
            process_ref_image(first_frame_path, target_width, target_height)
        )

    return ref_images_list, [len(ref_images_list)], obj_info_list


def load_sequence_masks(mask_folder, num_frames=49, target_width=832, target_height=480, num_objects=None):
    if not os.path.exists(mask_folder):
        return torch.zeros((1, 1, num_frames, 1, target_height, target_width))

    files = sorted(f for f in os.listdir(mask_folder) if f.endswith(".png"))
    if num_objects is None:
        loaded_frames = []
        for i in range(num_frames):
            file_name = files[i] if i < len(files) else (files[-1] if files else None)
            if file_name:
                mask_pil = Image.open(os.path.join(mask_folder, file_name)).convert("L")
                if mask_pil.size != (target_width, target_height):
                    mask_pil = mask_pil.resize((target_width, target_height), Image.NEAREST)
                loaded_frames.append((TF.to_tensor(mask_pil) > 0.5).float())
            else:
                loaded_frames.append(torch.zeros((1, target_height, target_width)))
        masks_stacked = torch.stack(loaded_frames, dim=0).unsqueeze(0).unsqueeze(1)
        return masks_stacked

    first_path = os.path.join(mask_folder, files[0]) if files else None
    unique_colors = []
    if first_path:
        first_np = np.array(Image.open(first_path).convert("RGB"))
        for color in np.unique(first_np.reshape(-1, 3), axis=0):
            if not np.all(color == 0):
                unique_colors.append(tuple(color))
    if len(unique_colors) < num_objects:
        unique_colors = [tuple(COLOR_PALETTE[i]) for i in range(num_objects)]
    unique_colors = unique_colors[:num_objects]

    all_objects_masks = []
    for obj_idx in range(num_objects):
        obj_color = unique_colors[obj_idx]
        obj_frames = []
        for i in range(num_frames):
            file_name = files[i] if i < len(files) else (files[-1] if files else None)
            if file_name:
                mask_np = np.array(
                    Image.open(os.path.join(mask_folder, file_name))
                    .convert("RGB")
                    .resize((target_width, target_height), Image.NEAREST)
                )
                color_diff = np.abs(
                    mask_np.astype(np.float32) - np.array(obj_color).astype(np.float32)
                )
                obj_mask = np.all(color_diff < 10, axis=-1).astype(np.float32)
                obj_frames.append(torch.from_numpy(obj_mask).unsqueeze(0))
            else:
                obj_frames.append(torch.zeros((1, target_height, target_width)))
        all_objects_masks.append(torch.stack(obj_frames, dim=0))

    return torch.stack(all_objects_masks, dim=0).unsqueeze(1)


def preprocess_bbox_frame(image, min_value=-1, max_value=1):
    img_np = np.array(image, dtype=np.float32)
    image_tensor = torch.from_numpy(img_np)
    image_tensor = image_tensor * ((max_value - min_value) / 255.0) + min_value
    return image_tensor.permute(2, 0, 1)


def generate_bbox_conditions(frames_info, num_frames, height, width, ordered_ids=None):
    if ordered_ids is None:
        all_obj_ids = set()
        for frame_data in frames_info:
            for bbox in frame_data.get("bboxes", []):
                all_obj_ids.add(str(bbox["obj_id"]))
        sorted_obj_ids = sorted(all_obj_ids, key=_obj_id_sort_key)
    else:
        sorted_obj_ids = [str(oid) for oid in ordered_ids]

    obj_id_to_idx = {oid: i for i, oid in enumerate(sorted_obj_ids)}
    num_objs = len(sorted_obj_ids) if sorted_obj_ids else 1

    bbox_frames_list = []
    object_bbox_masks_np = np.zeros((num_objs, num_frames, 1, height, width), dtype=np.uint8)
    obj_colors = {oid: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, oid in enumerate(sorted_obj_ids)}

    for t in range(num_frames):
        frame_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        if t < len(frames_info):
            for box_data in frames_info[t].get("bboxes", []):
                obj_id = str(box_data["obj_id"])
                if obj_id not in obj_id_to_idx:
                    continue
                obj_idx = obj_id_to_idx[obj_id]
                x1, y1, x2, y2 = map(int, box_data["bbox"])
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                frame_canvas[y1:y2, x1:x2, :] = obj_colors[obj_id]
                object_bbox_masks_np[obj_idx, t, 0, y1:y2, x1:x2] = 1
        bbox_frames_list.append(preprocess_bbox_frame(frame_canvas))

    bbox_mask_tensor = torch.stack(bbox_frames_list, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    object_bbox_masks_tensor = torch.from_numpy(object_bbox_masks_np).float()
    return bbox_mask_tensor, object_bbox_masks_tensor


def load_tracks_from_json(json_data, video_root_path, target_frames=49):
    track_info = json_data.get("paths", {}).get("tracks", {})
    if not track_info:
        return None, None

    track_path = os.path.join(video_root_path, track_info.get("tracks", ""))
    vis_path = os.path.join(video_root_path, track_info.get("visibility", ""))
    if not os.path.exists(track_path):
        return None, None

    tracks_np = np.load(track_path)
    if tracks_np.ndim == 3:
        tracks_np = tracks_np[np.newaxis, ...]

    if vis_path and os.path.exists(vis_path):
        vis_np = np.load(vis_path)
        if vis_np.ndim == 2:
            vis_np = vis_np[np.newaxis, ...]
        vis_np = vis_np.astype(bool)
    else:
        vis_np = np.ones(tracks_np.shape[:3], dtype=bool)

    curr_t = tracks_np.shape[1]
    if curr_t >= target_frames:
        tracks_np = tracks_np[:, :target_frames, :, :]
        vis_np = vis_np[:, :target_frames, :]
    else:
        pad_len = target_frames - curr_t
        tracks_np = np.concatenate(
            [tracks_np, np.repeat(tracks_np[:, -1:, :, :], pad_len, axis=1)], axis=1
        )
        vis_np = np.concatenate(
            [vis_np, np.repeat(vis_np[:, -1:, :], pad_len, axis=1)], axis=1
        )

    return torch.from_numpy(tracks_np).float(), torch.from_numpy(vis_np).bool()


def _load_metadata(case_folder):
    import json

    meta_path = os.path.join(case_folder, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def build_omni_conditions(case_folder, device, dtype, num_frames=49, height=480, width=832):
    data = _load_metadata(case_folder)
    mode = data.get("mode", "single_object")
    paths = data.get("paths", {})

    frames_info = data.get("frames_info") or []
    canon_ids = ordered_obj_ids(data) if (frames_info or paths.get("ref_imgs")) else []

    ref_images = []
    ref_indicators = []
    ref_info = paths.get("ref_imgs") or []
    if ref_info and canon_ids:
        ref_images, ref_indicators = load_refs_from_metadata(
            case_folder, data, canon_ids, width, height
        )
    elif mode == "multiple_object":
        ref_dir = os.path.join(case_folder, "reference_images")
        if os.path.isdir(ref_dir):
            ref_images, _, obj_info_list = extract_multi_ref_imgs(ref_dir, width, height)
            if canon_ids:
                id_to_img = {str(oid): img for (oid, _, _), img in zip(obj_info_list, ref_images)}
                ref_images = [id_to_img[oid] for oid in canon_ids if oid in id_to_img]
            ref_indicators = [len(ref_images)] if ref_images else []

    num_objs = len(ref_images) if mode == "multiple_object" and ref_images else None

    if frames_info:
        bbox_mask, object_bbox_masks = generate_bbox_conditions(
            frames_info, num_frames, height, width, ordered_ids=canon_ids or None
        )
        bbox_mask = bbox_mask.to(device, dtype)
        object_bbox_masks = object_bbox_masks.to(device, dtype)
    else:
        bbox_mask = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=dtype)
        object_bbox_masks = torch.zeros((1, num_frames, 1, height, width), device=device, dtype=dtype)

    masks_rel = paths.get("masks")
    masks_dir = os.path.join(case_folder, masks_rel) if masks_rel else ""
    if masks_rel and os.path.isdir(masks_dir):
        object_masks = load_sequence_masks(
            masks_dir, num_frames, width, height, num_objects=num_objs
        ).to(device, dtype)
    elif frames_info:
        object_masks = object_bbox_masks.unsqueeze(0).to(device, dtype)
    else:
        object_masks = None

    tracks, vis = load_tracks_from_json(data, case_folder, num_frames)
    if tracks is not None:
        tracks = tracks.to(device)
        vis = vis.to(device)

    num_obj_slots = int(object_bbox_masks.shape[0]) if frames_info else len(ref_images)
    if not num_obj_slots and tracks is not None:
        num_obj_slots = tracks.shape[2]

    track_channel_obj_slot = None
    if tracks is not None and data.get("obj_track_idx") is not None:
        n_tracks = tracks.shape[2]
        track_channel_obj_slot = build_track_channel_obj_slot(data, n_tracks, num_obj_slots)

    if not ref_indicators and num_obj_slots:
        ref_indicators = [num_obj_slots]

    return {
        "data": data,
        "ref_images": ref_images,
        "ref_indicators": ref_indicators,
        "object_masks": object_masks,
        "tracks": tracks,
        "vis": vis,
        "bbox_mask": bbox_mask,
        "object_bbox_masks": object_bbox_masks,
        "track_channel_obj_slot": track_channel_obj_slot,
        "caption": data.get("caption", ""),
    }
