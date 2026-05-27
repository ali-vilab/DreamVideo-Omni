import torch
from PIL import Image, ImageDraw
from diffsynth.data.video import save_video
from diffsynth.pipelines.wan_video_omni import WanVideoPipeline, ModelConfig, WanVideoUnit_OmniVideoEmbedder_Inference
import numpy as np
import argparse
import torchvision.transforms.functional as TF
import cv2
import json
import os
from checkpoint_utils import load_checkpoint_to_pipeline
from omni_case_loader import build_omni_conditions
import ssl
import itertools
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs

                                   
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

                                          
COLOR_PALETTE = [
    [255, 0, 0],       
    [0, 255, 0],       
    [0, 0, 255],       
    [255, 255, 0],     
    [255, 0, 255],     
    [0, 255, 255],     
]

                                           
def draw_overall_gradient_polyline_on_image(image, line_width, points, start_color):
    def get_distance(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    new_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(new_image, 'RGBA')
    points = points[::-1]

    if len(points) < 2: return new_image
    total_length = sum(get_distance(points[i], points[i+1]) for i in range(len(points)-1))
    if total_length == 0: return new_image

    accumulated_length = 0
    for start_point, end_point in zip(points[:-1], points[1:]):
        segment_length = get_distance(start_point, end_point)
        steps = int(max(segment_length, 1))
        for i in range(steps):
            current_length = accumulated_length + (i / steps) * segment_length
            alpha = int(255 * (1 - current_length / total_length))
            color = (*start_color, alpha)
            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)
            dynamic_line_width = int(line_width * (1 - (current_length / total_length)))
            dynamic_line_width = max(dynamic_line_width, 1)
            offset = dynamic_line_width / 2
            draw.ellipse((x - offset, y - offset, x + offset, y + offset), fill=color)
        accumulated_length += segment_length
    return new_image

def add_weighted(rgb, track):
    rgb = np.array(rgb)
    track = np.array(track)
    alpha = track[:, :, 3] / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    blend_img = track[:, :, :3] * alpha + rgb * (1 - alpha)
    return Image.fromarray(blend_img.astype(np.uint8))

def draw_tracks_on_video(video_frames_pil, tracks_tensor, vis_tensor=None, track_frame=21):
    color_map = [
        (102, 153, 255), (0, 255, 255), (255, 255, 0), (255, 102, 204),
        (0, 255, 0), (255, 0, 0), (128, 0, 128), (255, 165, 0),
        (255, 255, 255), (165, 42, 42)
    ]
    circle_size = 8
    line_width = 12

    if tracks_tensor is None: return video_frames_pil

    tracks_np = tracks_tensor.float().cpu().numpy()
    if tracks_np.ndim == 4: tracks_np = tracks_np[0]
    tracks_np = np.transpose(tracks_np, (1, 0, 2))

    vis_np = None
    if vis_tensor is not None:
        vis_np = vis_tensor.cpu().numpy()
        if vis_np.ndim == 3: vis_np = vis_np[0]
        vis_np = np.transpose(vis_np, (1, 0))

    output_frames = []
    num_frames = len(video_frames_pil)
    num_tracks = tracks_np.shape[0]

    for t in range(num_frames):
        frame_pil = video_frames_pil[t].copy()
        for n in range(num_tracks):
            if vis_np is not None:
                if t < vis_np.shape[1] and not vis_np[n, t]:
                    continue

            if t >= tracks_np.shape[1]: continue

            pt = tracks_np[n, t]
            start_t = max(t - track_frame, 0)
            history_pts = tracks_np[n, start_t:t+1]
            points_list = [(float(p[0]), float(p[1])) for p in history_pts]
            color = color_map[n % len(color_map)]

            if len(points_list) > 1:
                track_layer = draw_overall_gradient_polyline_on_image(frame_pil, line_width, points_list, color)
                frame_pil = add_weighted(frame_pil, track_layer)

            draw = ImageDraw.Draw(frame_pil)
            draw.ellipse((pt[0] - circle_size, pt[1] - circle_size, pt[0] + circle_size, pt[1] + circle_size), fill=color)
        output_frames.append(frame_pil)
    return output_frames

def draw_bboxes_on_video(video_frames_pil, frames_info, width, height):
    output_frames = []
    for t, frame_pil in enumerate(video_frames_pil):
        frame_cv = np.array(frame_pil)
        if t < len(frames_info):
            frame_data = frames_info[t]
            for bbox_info in frame_data.get('bboxes', []):
                bbox = bbox_info['bbox']
                obj_id = int(bbox_info['obj_id'])
                x1, y1, x2, y2 = map(int, bbox)
                color = COLOR_PALETTE[obj_id % len(COLOR_PALETTE)]
                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), color, 2)
                label = bbox_info.get('label', f'obj{obj_id}')
                cv2.putText(frame_cv, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        output_frames.append(Image.fromarray(frame_cv))
    return output_frames

                                            

def simple_bbox_to_model_input(bboxes_list, num_frames, height, width, device, dtype):
    def preprocess_bbox_frame(image, min_value=-1, max_value=1):
        img_np = np.array(image, dtype=np.float32)
        image_tensor = torch.from_numpy(img_np)
        image_tensor = image_tensor * ((max_value - min_value) / 255.0) + min_value
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

    bbox_frames_list = []
    object_bbox_masks_np = np.zeros((1, num_frames, 1, height, width), dtype=np.uint8)
    color = COLOR_PALETTE[0]

    for t in range(num_frames):
        frame_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        if t < len(bboxes_list):
            bbox = bboxes_list[t]
            x1, y1, x2, y2 = map(int, bbox)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                frame_canvas[y1:y2, x1:x2, :] = color
                object_bbox_masks_np[0, t, 0, y1:y2, x1:x2] = 1
        processed_frame = preprocess_bbox_frame(frame_canvas)
        bbox_frames_list.append(processed_frame)

    bbox_mask_tensor = torch.stack(bbox_frames_list, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    object_bbox_masks_tensor = torch.from_numpy(object_bbox_masks_np).float()
    return bbox_mask_tensor.to(device, dtype), object_bbox_masks_tensor.to(device, dtype)

def simple_trajectory_to_model_input(trajectory_list, num_frames, device):
    tracks_np = np.zeros((1, num_frames, 1, 2), dtype=np.float32)
    vis_np = np.ones((1, num_frames, 1), dtype=bool)
    for t in range(min(len(trajectory_list), num_frames)):
        point = trajectory_list[t]
        tracks_np[0, t, 0, :] = [point[0], point[1]]
    tracks_tensor = torch.from_numpy(tracks_np).float().to(device)
    vis_tensor = torch.from_numpy(vis_np).bool().to(device)
    return tracks_tensor, vis_tensor

def multi_bbox_to_model_input(bboxes_list_per_obj, num_frames, height, width, device, dtype):
    def preprocess_bbox_frame(image, min_value=-1, max_value=1):
        img_np = np.array(image, dtype=np.float32)
        image_tensor = torch.from_numpy(img_np)
        image_tensor = image_tensor * ((max_value - min_value) / 255.0) + min_value
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

    num_objs = len(bboxes_list_per_obj)
    bbox_frames_list = []
    object_bbox_masks_np = np.zeros((num_objs, num_frames, 1, height, width), dtype=np.uint8)

    for t in range(num_frames):
        frame_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        for obj_idx, bboxes_list in enumerate(bboxes_list_per_obj):
            if t < len(bboxes_list):
                bbox = bboxes_list[t]
                x1, y1, x2, y2 = map(int, bbox)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 > x1 and y2 > y1:
                    color = COLOR_PALETTE[obj_idx % len(COLOR_PALETTE)]
                    frame_canvas[y1:y2, x1:x2, :] = color
                    object_bbox_masks_np[obj_idx, t, 0, y1:y2, x1:x2] = 1
        processed_frame = preprocess_bbox_frame(frame_canvas)
        bbox_frames_list.append(processed_frame)

    bbox_mask_tensor = torch.stack(bbox_frames_list, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    object_bbox_masks_tensor = torch.from_numpy(object_bbox_masks_np).float()
    return bbox_mask_tensor.to(device, dtype), object_bbox_masks_tensor.to(device, dtype)

def trajectory_to_object_masks(trajectory_list_per_obj, num_frames, height, width, radius=32):
    num_objs = len(trajectory_list_per_obj)
    masks = np.zeros((num_objs, num_frames, 1, height, width), dtype=np.uint8)
    for obj_idx, traj in enumerate(trajectory_list_per_obj):
        for t in range(min(len(traj), num_frames)):
            cx, cy = int(traj[t][0]), int(traj[t][1])
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(width, cx + radius), min(height, cy + radius)
            masks[obj_idx, t, 0, y1:y2, x1:x2] = 1
    return torch.from_numpy(masks).float()


def multi_trajectory_to_model_input(trajectory_list_per_obj, num_frames, device):
    num_objs = len(trajectory_list_per_obj)
    tracks_np = np.zeros((1, num_frames, num_objs, 2), dtype=np.float32)
    vis_np = np.ones((1, num_frames, num_objs), dtype=bool)
    for obj_idx, trajectory_list in enumerate(trajectory_list_per_obj):
        for t in range(min(len(trajectory_list), num_frames)):
            point = trajectory_list[t]
            tracks_np[0, t, obj_idx, :] = [point[0], point[1]]
    tracks_tensor = torch.from_numpy(tracks_np).float().to(device)
    vis_tensor = torch.from_numpy(vis_np).bool().to(device)
    return tracks_tensor, vis_tensor

                                             

def process_and_align_ref_image(image_path, target_bbox, canvas_w, canvas_h, white_threshold=240):
    if not os.path.exists(image_path):
        print(f"[Error] Reference image not found: {image_path}")
        return Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    pil_image = Image.open(image_path).convert('RGB')
    
                           
    base_name, ext = os.path.splitext(image_path)
    mask_path_1 = f"{base_name}_mask{ext}" 
    mask_path_2 = f"{base_name}_mask.png"   
    
    final_mask = None
    
    if os.path.exists(mask_path_2):
        final_mask = Image.open(mask_path_2).convert("L")
    elif os.path.exists(mask_path_1):
        final_mask = Image.open(mask_path_1).convert("L")
    
    img_rgba = pil_image.convert("RGBA")
    
    if final_mask is not None:
        if final_mask.size != pil_image.size:
            final_mask = final_mask.resize(pil_image.size, Image.NEAREST)
        img_rgba.putalpha(final_mask)
    else:
                                           
        img_np = np.array(pil_image)
        is_white = np.all(img_np > white_threshold, axis=-1)
        alpha_channel = np.where(is_white, 0, 255).astype(np.uint8)
        img_rgba.putalpha(Image.fromarray(alpha_channel))

    content_bbox = img_rgba.getbbox()
    if content_bbox:
        cropped_img = img_rgba.crop(content_bbox)
    else:
        cropped_img = img_rgba
    
    src_w, src_h = cropped_img.size
    
                      
    if target_bbox is None or len(target_bbox) < 4:
                            
         tgt_x1, tgt_y1, tgt_x2, tgt_y2 = 0, 0, canvas_w, canvas_h
    else:
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = map(int, target_bbox)
        
    tgt_w = max(1, tgt_x2 - tgt_x1)
    tgt_h = max(1, tgt_y2 - tgt_y1)
    
    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    
    resized_img = cropped_img.resize((new_w, new_h), Image.LANCZOS)
    
    center_x = (tgt_x1 + tgt_x2) / 2
    center_y = (tgt_y1 + tgt_y2) / 2
    paste_x = int(center_x - new_w / 2)
    paste_y = int(center_y - new_h / 2)
    
    final_canvas_rgb = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    final_canvas_rgb.paste(resized_img, (paste_x, paste_y), mask=resized_img)
    
    return final_canvas_rgb

def process_ref_image_simple(image_path, target_width=832, target_height=480):
    if not os.path.exists(image_path):
        return Image.new("RGB", (target_width, target_height), (0, 0, 0))
    pil_image = Image.open(image_path).convert('RGB')
    final_img = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return final_img

                                               

def get_image_list_from_path(path_str):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_list = []
    
    path_str = os.path.expanduser(path_str)
    
    if os.path.isfile(path_str):
        image_list.append(path_str)
    elif os.path.isdir(path_str):
        files = sorted(os.listdir(path_str)) 
        for f in files:
            if f.startswith('.'): continue 
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_exts:
                if not name.endswith('_mask') and not name.endswith('_vis'):
                    image_list.append(os.path.join(path_str, f))
    else:
        print(f"[Warning] Path not found: {path_str}")
        
    return image_list

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def init_pipeline(args, device):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
        local_model_path=args.local_model_path,
        skip_download=args.skip_download,
        inject_bbox_per_layer=True,
    )
    for i, unit in enumerate(pipe.units):
        if "OmniVideoEmbedder" in str(type(unit)):
            pipe.units[i] = WanVideoUnit_OmniVideoEmbedder_Inference()
    load_checkpoint_to_pipeline(pipe, args.checkpoint, device="cpu")
    pipe.to(device)
    return pipe


def _infer_from_conditions(args, pipe, accelerator, device, seed, cond, case_folder=None):
    seed_list = getattr(args, "seed_list", None)
    if seed_list:
        if accelerator.process_index >= len(seed_list):
            accelerator.wait_for_everyone()
            return
        seed = seed_list[accelerator.process_index]
        base, ext = os.path.splitext(args.output_path)
        output_path = f"{base}_seed{seed}{ext or '.mp4'}"
    elif accelerator.process_index != 0:
        accelerator.wait_for_everyone()
        return
    else:
        output_path = args.output_path

    video_output = pipe(
        prompt=[cond["caption"]],
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=args.num_inference_steps,
        tiled=True,
        reference_imgs=cond["ref_images"],
        reference_imgs_indicator=cond["ref_indicators"],
        object_masks=cond["object_masks"],
        input_tracks=cond["tracks"],
        input_visibilities=cond["vis"],
        bbox_mask=cond["bbox_mask"],
        object_bbox_masks=cond["object_bbox_masks"],
        track_channel_obj_slot=cond.get("track_channel_obj_slot"),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=seed,
    )
    gen_frames = video_output[0] if isinstance(video_output, (list, tuple)) else video_output
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_video(gen_frames, output_path, fps=15, quality=5)
    print(f"[Rank {accelerator.process_index}] saved {output_path}")

    if case_folder and cond.get("data", {}).get("frames_info"):
        base, ext = os.path.splitext(output_path)
        vis_path = f"{base}_vis{ext or '.mp4'}"
        vis_frames = gen_frames
        if cond["tracks"] is not None:
            vis_frames = draw_tracks_on_video(vis_frames, cond["tracks"], cond["vis"])
        vis_frames = draw_bboxes_on_video(
            vis_frames, cond["data"]["frames_info"], args.width, args.height
        )
        save_video(vis_frames, vis_path, fps=15, quality=5)

    accelerator.wait_for_everyone()


def run_case_dir(args, pipe, accelerator, device, seed):
    case_folder = os.path.abspath(args.case_dir)
    meta_path = os.path.join(case_folder, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {case_folder}")

    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)

    mode = data.get("mode", "single_object")
    paths = data.get("paths", {})
    has_tracks = bool(paths.get("tracks"))
    has_frames = bool(data.get("frames_info"))
    has_ref = bool(paths.get("ref_imgs"))

    if accelerator.is_main_process:
        print(
            f"[Info] case_dir={case_folder} mode={mode} "
            f"ref={has_ref} tracks={has_tracks} bbox_frames={has_frames}"
        )

    if has_tracks or has_frames:
        cond = build_omni_conditions(
            case_folder, device, pipe.torch_dtype,
            num_frames=args.num_frames, height=args.height, width=args.width,
        )
        _infer_from_conditions(args, pipe, accelerator, device, seed, cond, case_folder)
        return

    ref_entries = paths.get("ref_imgs", [])
    if not ref_entries:
        raise ValueError(f"No controls in {meta_path} (need ref_imgs and/or tracks/frames_info)")
    ref_paths = [os.path.join(case_folder, r["path"]) for r in ref_entries]
    for p in ref_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Reference image not found: {p}")

    manual = argparse.Namespace(**vars(args))
    manual.mode = mode
    manual.prompt = data.get("caption", "")
    manual.ref_images = ",".join(ref_paths)
    manual.bbox_files = None
    manual.trajectory_files = None
    manual.auto_position_ref = True
    run_manual_infer(manual, pipe, accelerator, device, seed)


def run_manual_infer(args, pipe, accelerator, device, seed):
    has_ref = bool(args.ref_images and args.ref_images.strip())
    has_bbox = bool(args.bbox_files and args.bbox_files.strip())
    has_traj = bool(args.trajectory_files and args.trajectory_files.strip())
    if not (has_ref or has_bbox or has_traj):
        raise ValueError("At least one of --ref_images, --bbox_files, or --trajectory_files is required.")

    bbox_paths = [p.strip() for p in args.bbox_files.split(",")] if has_bbox else []
    traj_paths = [p.strip() for p in args.trajectory_files.split(",")] if has_traj else []

    object_image_pools = []
    if has_ref:
        input_paths = [p.strip() for p in args.ref_images.split(",")]
        for idx, p in enumerate(input_paths):
            imgs = get_image_list_from_path(p)
            if not imgs:
                raise ValueError(f"No valid images found in path: {p}")
            object_image_pools.append(imgs)
            if accelerator.is_main_process:
                print(f"[Info] Object slot {idx}: {len(imgs)} images in {p}")
    else:
        n_slots = max(len(bbox_paths), len(traj_paths), 1)
        object_image_pools = [[None]] * n_slots

    if has_bbox and has_ref and len(bbox_paths) != len(object_image_pools):
        raise ValueError(f"{len(object_image_pools)} ref slot(s) vs {len(bbox_paths)} bbox file(s).")
    if has_traj and has_ref and len(traj_paths) != len(object_image_pools):
        raise ValueError(f"{len(object_image_pools)} ref slot(s) vs {len(traj_paths)} trajectory file(s).")
    if has_bbox and has_traj and len(bbox_paths) != len(traj_paths):
        raise ValueError(f"{len(bbox_paths)} bbox file(s) vs {len(traj_paths)} trajectory file(s).")

    combinations = list(itertools.product(*object_image_pools))
    if accelerator.is_main_process:
        print(f"[Info] Combinations to generate: {len(combinations)}")
    seed_list = getattr(args, "seed_list", None)
    if seed_list:
        if accelerator.process_index >= len(seed_list):
            my_combinations = []
        elif combinations:
            my_combinations = [combinations[0]]
        else:
            my_combinations = [tuple([None] * len(object_image_pools))]
    else:
        my_combinations = combinations[accelerator.process_index::accelerator.num_processes]

    enable_bbox_condition = False
    bboxes_list_per_obj = []
    if has_bbox:
        from simple_interpolation import auto_interpolate_bboxes
        enable_bbox_condition = True
        for bp in bbox_paths:
            if not os.path.exists(bp):
                raise FileNotFoundError(bp)
            input_bboxes = []
            with open(bp) as f:
                for line in f:
                    if line.strip():
                        input_bboxes.append([float(x) for x in line.strip().split(",")])
            if len(input_bboxes) < args.num_frames:
                bboxes_list_per_obj.append(auto_interpolate_bboxes(input_bboxes, args.num_frames))
            else:
                bboxes_list_per_obj.append(input_bboxes[:args.num_frames])

    enable_traj_condition = False
    trajectory_list_per_obj = []
    if has_traj:
        from simple_interpolation import auto_interpolate_trajectory
        enable_traj_condition = True
        for tp in traj_paths:
            if not os.path.exists(tp):
                raise FileNotFoundError(tp)
            input_points = []
            with open(tp) as f:
                for line in f:
                    if line.strip():
                        input_points.append([float(x) for x in line.strip().split(",")])
            if len(input_points) < args.num_frames:
                trajectory_list_per_obj.append(auto_interpolate_trajectory(input_points, args.num_frames))
            else:
                trajectory_list_per_obj.append(input_points[:args.num_frames])

    bbox_mask_tensor = None
    object_bbox_masks_tensor = None
    object_masks_tensor = None
    tracks_tensor = None
    vis_tensor = None

    if enable_bbox_condition:
        if args.mode == "single_object":
            bbox_mask_tensor, object_bbox_masks_tensor = simple_bbox_to_model_input(
                bboxes_list_per_obj[0], args.num_frames, args.height, args.width, device, pipe.torch_dtype
            )
        else:
            bbox_mask_tensor, object_bbox_masks_tensor = multi_bbox_to_model_input(
                bboxes_list_per_obj, args.num_frames, args.height, args.width, device, pipe.torch_dtype
            )
        object_masks_tensor = object_bbox_masks_tensor.unsqueeze(0)

    if enable_traj_condition and not enable_bbox_condition:
        traj_masks = trajectory_to_object_masks(
            trajectory_list_per_obj, args.num_frames, args.height, args.width
        )
        object_masks_tensor = traj_masks.unsqueeze(0).to(device)

    if enable_traj_condition:
        if args.mode == "single_object":
            tracks_tensor, vis_tensor = simple_trajectory_to_model_input(
                trajectory_list_per_obj[0], args.num_frames, device
            )
        else:
            tracks_tensor, vis_tensor = multi_trajectory_to_model_input(
                trajectory_list_per_obj, args.num_frames, device
            )

    base_output_name, ext = os.path.splitext(args.output_path)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    num_objects = max(
        len(bboxes_list_per_obj) if enable_bbox_condition else 0,
        len(trajectory_list_per_obj) if enable_traj_condition else 0,
        len(object_image_pools),
        1,
    )

    for i, ref_imgs_combo in enumerate(my_combinations):
        processed_ref_images = []
        combo_names = []
        if ref_imgs_combo:
            total_objs = len(ref_imgs_combo)
            for idx, ref_path in enumerate(ref_imgs_combo):
                if ref_path is None:
                    continue
                combo_names.append(os.path.splitext(os.path.basename(ref_path))[0])
                if enable_bbox_condition and idx < len(bboxes_list_per_obj):
                    aligned_img = process_and_align_ref_image(
                        ref_path, bboxes_list_per_obj[idx][0], args.width, args.height
                    )
                elif args.auto_position_ref:
                    slot_width = args.width // total_objs
                    auto_bbox = [idx * slot_width, 0, (idx + 1) * slot_width, args.height]
                    aligned_img = process_and_align_ref_image(ref_path, auto_bbox, args.width, args.height)
                else:
                    aligned_img = process_ref_image_simple(ref_path, args.width, args.height)
                processed_ref_images.append(aligned_img)
        else:
            combo_names.append("no_ref")

        if processed_ref_images:
            ref_indicators = [1] if args.mode == "single_object" else [len(processed_ref_images)]
        else:
            ref_indicators = [1] if args.mode == "single_object" else [num_objects]

        if seed_list:
            current_output_path = f"{base_output_name}_seed{seed}{ext or '.mp4'}"
        elif len(my_combinations) == 1:
            current_output_path = args.output_path
        else:
            current_output_path = f"{base_output_name}_{'_'.join(combo_names)}{ext}"

        video_output = pipe(
            prompt=[args.prompt],
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.num_inference_steps,
            tiled=True,
            reference_imgs=processed_ref_images or None,
            reference_imgs_indicator=ref_indicators,
            object_masks=object_masks_tensor,
            input_tracks=tracks_tensor,
            input_visibilities=vis_tensor,
            bbox_mask=bbox_mask_tensor,
            object_bbox_masks=object_bbox_masks_tensor,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=seed,
        )

        gen_frames = video_output[0] if isinstance(video_output, (list, tuple)) else video_output
        save_video(gen_frames, current_output_path, fps=15, quality=5)
        print(f"[Rank {accelerator.process_index}] saved {current_output_path}")

        if enable_bbox_condition or enable_traj_condition:
            vis_path = current_output_path.replace(".mp4", "_vis.mp4")
            vis_frames = gen_frames
            if enable_traj_condition:
                vis_frames = draw_tracks_on_video(vis_frames, tracks_tensor, vis_tensor)
            if enable_bbox_condition:
                frames_info = []
                for t in range(args.num_frames):
                    frame_data = {"bboxes": []}
                    for obj_idx, bboxes_list in enumerate(bboxes_list_per_obj):
                        if t < len(bboxes_list):
                            frame_data["bboxes"].append(
                                {"bbox": bboxes_list[t], "label": "obj", "obj_id": str(obj_idx)}
                            )
                    frames_info.append(frame_data)
                vis_frames = draw_bboxes_on_video(vis_frames, frames_info, args.width, args.height)
            save_video(vis_frames, vis_path, fps=15, quality=5)


def main():
    parser = argparse.ArgumentParser(
        description="DreamVideo-Omni inference: examples/<case> or manual ref/bbox/trajectory"
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["single_object", "multiple_object"], default="single_object")
    
    parser.add_argument("--ref_images", type=str, default=None, help="Optional. Image file or directory per object, comma-separated.")
    
                                         
    parser.add_argument("--auto_position_ref", action="store_true", help="If True and no bbox file is provided, automatically distribute reference images horizontally.")

    parser.add_argument("--bbox_files", type=str, default=None)
    parser.add_argument("--trajectory_files", type=str, default=None)
    
           
    parser.add_argument("--ref_image", type=str, default=None) 
    parser.add_argument("--bbox_file", type=str, default=None)
    parser.add_argument("--trajectory_file", type=str, default=None)

    parser.add_argument("--output_path", type=str, default="output_demo.mp4")
    parser.add_argument(
        "--case_dir",
        type=str,
        default=None,
        help="Single example folder (metadata.json + reference/mask/track assets). "
             "Preferred for bundled examples under examples/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/dreamvideo_omni_sft.safetensors",
        help="DreamVideo-Omni DiT (SFT) weights, e.g. dreamvideo_omni_sft.safetensors (.safetensors, .pth, or .bin).",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="./models",
        help="Root directory for Wan base weights. Files are read from "
             "<local_model_path>/Wan-AI/Wan2.1-T2V-1.3B/ when present; otherwise downloaded from ModelScope.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Do not download Wan base weights; fail if files are missing under --local_model_path.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-GPU runs (one seed per rank). "
             "Use with: accelerate launch --num_processes N infer.py ...",
    )
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    if args.mode == "single_object":
        if args.ref_image and not args.ref_images:
            args.ref_images = args.ref_image
        if args.bbox_file and not args.bbox_files:
            args.bbox_files = args.bbox_file
        if args.trajectory_file and not args.trajectory_files:
            args.trajectory_files = args.trajectory_file

    use_case_dir = bool(args.case_dir)
    if not use_case_dir and not args.prompt:
        raise ValueError("--prompt is required unless --case_dir is set.")

    args.seed_list = None
    if args.seeds:
        args.seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=48))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device

    if args.seed_list:
        if accelerator.process_index >= len(args.seed_list):
            if accelerator.is_main_process:
                print(f"[Info] rank>={len(args.seed_list)}, nothing to do.")
            accelerator.wait_for_everyone()
            return
        seed = args.seed_list[accelerator.process_index]
    else:
        seed = args.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)

    if accelerator.is_main_process:
        mode_name = "case_dir" if use_case_dir else "manual"
        seed_info = args.seed_list if args.seed_list else seed
        print(f"[Info] mode={args.mode} run={mode_name} gpus={accelerator.num_processes} seeds={seed_info}")

    if accelerator.is_main_process:
        print("[Info] Initializing pipeline...")
    pipe = init_pipeline(args, device)

    if use_case_dir:
        run_case_dir(args, pipe, accelerator, device, seed)
    else:
        run_manual_infer(args, pipe, accelerator, device, seed)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("[Info] All ranks finished.")

if __name__ == "__main__":
    main()