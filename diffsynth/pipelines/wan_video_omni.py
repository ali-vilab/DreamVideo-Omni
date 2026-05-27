import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ..utils.track_utils import create_pos_feature_map
from contextlib import nullcontext

class ZeroConv1D(torch.nn.Module):
    def __init__(self, channels: int, dtype=torch.bfloat16):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels, channels, kernel_size=1, dtype=dtype)
        
        torch.nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
                               
        return self.conv(x)

class ZeroConv3D(torch.nn.Module):
    def __init__(self, channels: int, dtype=torch.bfloat16):
        super().__init__()
        self.conv = torch.nn.Conv3d(channels, channels, kernel_size=1, dtype=dtype)
              
        torch.nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
                                           
        return self.conv(x)

class MLPConv3D(nn.Module):
    def __init__(self, channels: int, hidden_channels: list = [32], dtype=torch.bfloat16):
        super().__init__()
        layers = []
        in_ch = channels
        
               
        for h_ch in hidden_channels:
            layers.append(nn.Conv3d(in_ch, h_ch, kernel_size=1, dtype=dtype))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = h_ch
        
                         
        final_conv = nn.Conv3d(in_ch, channels, kernel_size=1, dtype=dtype)
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
                                           
        return self.mlp(x)

class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
                                                                                                   
        self.device = device
        self.torch_dtype = torch_dtype
                                                            
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
                     
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
                                               
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
                                                       
                                                                                                                                                     
                                                                 
        video = video.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return video

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
                                               
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

                                                                                                
                                                         
                                  
                                                                                        
                                                                                                                                      
                      

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        videos = [
            [
                self.vae_output_to_image(frame, pattern="H W C", min_value=min_value, max_value=max_value)
                for frame in sample.permute(1, 2, 3, 0)                        
            ]
            for sample in vae_output                               
        ]
        return videos

    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
                            
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
                           
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
                                   
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def enable_cpu_offload(self):
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")
        self.vram_management_enabled = True
        
        
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)
    
    
    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)

@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    skip_download: bool = False

    def _model_root(self, local_model_path):
        return os.path.join(local_model_path, self.model_id)

    def _local_files_ready(self, local_model_path):
        model_root = self._model_root(local_model_path)
        if not os.path.isdir(model_root):
            return False
        if self.origin_file_pattern is None or self.origin_file_pattern == "":
            return len(os.listdir(model_root)) > 0
        pattern = os.path.join(model_root, self.origin_file_pattern)
        return len(glob.glob(pattern)) > 0

    def download_if_necessary(self, local_model_path="./models", skip_download=False, use_usp=False):
        if self.path is None:
            if self.model_id is None:
                raise ValueError(
                    "No valid model files. Use ModelConfig(path='...') or "
                    "ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B', origin_file_pattern='...')."
                )

            if use_usp:
                import torch.distributed as dist
                skip_download = dist.get_rank() != 0

            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False

            model_root = self._model_root(local_model_path)
            files_ready = self._local_files_ready(local_model_path)
            skip_download = skip_download or self.skip_download or files_ready

            if files_ready:
                print(f"[Info] Using local Wan weights under {model_root}")
            elif not skip_download:
                print(f"[Info] Downloading {self.model_id} to {model_root}")
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=model_root) if os.path.isdir(model_root) else []
                snapshot_download(
                    self.model_id,
                    local_dir=model_root,
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=downloaded_files,
                    local_files_only=False,
                )
            else:
                print(f"[Info] skip_download=True, expecting weights under {model_root}")

            if use_usp:
                import torch.distributed as dist
                dist.barrier(device_ids=[dist.get_rank()])

            if is_folder:
                self.path = os.path.join(model_root, self.origin_file_pattern)
            else:
                self.path = glob.glob(os.path.join(model_root, self.origin_file_pattern))
            if isinstance(self.path, list) and len(self.path) == 0:
                raise FileNotFoundError(
                    f"No files matched {self.origin_file_pattern} under {model_root}. "
                    f"Set --local_model_path to the parent directory that contains Wan-AI/."
                )
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
                                                                         
        self.in_iteration_models = ("dit",)
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
                                                
            WanVideoUnit_OmniVideoEmbedder_Inference(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedder(),
                                        
                                          
                                              
                                          
                                  
            WanVideoUnit_UnifiedSequenceParallel(),
                                      
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video

    def set_omni_params(self, torch_dtype=torch.bfloat16, inject_bbox_per_layer=False, not_use_special_embedding=False, not_use_special_rope=False):
                                  
                                                                                       
        self.dit.bbox_zeroconv = ZeroConv3D(self.vae.z_dim, dtype=torch_dtype)
        self.inject_bbox_per_layer = inject_bbox_per_layer
        if self.inject_bbox_per_layer:
            self.dit.bbox_zeroconv_layers = nn.ModuleList(
                [ZeroConv1D(self.dit.dim, dtype=torch_dtype) for _ in range(self.dit.num_layers)]
            )
        self.downsample_ratios = [4, 8, 8]
        max_instances = 20

        self.use_special_embedding = not not_use_special_embedding
        self.use_special_rope = not not_use_special_rope
        if self.use_special_embedding:
                      
                                                                                 
                                                                                         
            self.dit.object_embed = torch.nn.Parameter(torch.randn(self.vae.z_dim))
            self.dit.control_signal_embed = torch.nn.Parameter(torch.randn(self.vae.z_dim))
                        
                                                                                     
            self.dit.group_id_embed = torch.nn.Embedding(max_instances, self.vae.z_dim)
        else:
            self.dit.object_embed = torch.zeros(self.vae.z_dim)
            self.dit.control_signal_embed = torch.zeros(self.vae.z_dim)
            self.dit.group_id_embed = torch.zeros(max_instances, self.vae.z_dim)

    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
        **omni_kwargs
    ):
                             
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
                             
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
                                  
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
                     
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

                              
        tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
                                   
        if use_usp: pipe.enable_usp()

        pipe.set_omni_params(torch_dtype=torch_dtype, **omni_kwargs)
        return pipe

    @torch.no_grad()
    def __call__(
        self,
                
        prompt: str,
        negative_prompt: Optional[str] = "",
                        
        input_image: Optional[Image.Image] = None,
                                   
        end_image: Optional[Image.Image] = None,
                        
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
                    
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
                        
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
              
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
                    
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
               
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
                                  
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
                   
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
                       
        motion_bucket_id: Optional[int] = None,
                    
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
                        
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
                  
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
                      
        progress_bar_cmd=tqdm,
        reference_imgs=None,
        reference_imgs_indicator=None,
        ref_imgs_latents_per_sample=None,
        bbox_mask=None,
        bbox_latents=None,
        track_video=None,
        object_bbox_masks=None,
        object_masks=None,
        input_tracks=None,       
        input_visibilities=None,
        track_channel_obj_slot=None,
    ):
                   
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
                
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": [negative_prompt] * len(prompt),
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
                                       
            "bbox_mask": bbox_mask,
            "reference_imgs": reference_imgs,
            "reference_imgs_indicator": reference_imgs_indicator,
            "object_masks": object_masks,
            "object_bbox_masks": object_bbox_masks,
            "track_video": track_video,
            "bbox_latents": bbox_latents, "ref_imgs_latents_per_sample": ref_imgs_latents_per_sample,
            "input_tracks": input_tracks, "input_visibilities": input_visibilities,
            "track_channel_obj_slot": track_channel_obj_slot,
                    }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

                 
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

                       
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

                       
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
                   
        
                                
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

                
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video

class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(self, pipe: WanVideoPipeline, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")

class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: WanVideoPipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
                                                            
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
                           
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
                           
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega

class WanVideoUnit_OmniVideoEmbedder_Inference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video", "noise", "tiled", "tile_size", "tile_stride", 
                "num_frames", "bbox_mask", "reference_imgs", 
                "reference_imgs_indicator", "object_bbox_masks", 
                "video_rgb", "object_masks", "height", "width",
                "input_tracks", "input_visibilities", "track_channel_obj_slot",
            ),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, num_frames, 
                bbox_mask, reference_imgs, reference_imgs_indicator, object_bbox_masks, 
                video_rgb, object_masks, height, width, input_tracks, input_visibilities,
                track_channel_obj_slot=None):
        pipe.load_models_to_device(["vae"])

                      
        use_bbox_cond = True
        if bbox_mask is None:
            use_bbox_cond = False
            bbox_mask = torch.zeros((1, 3, num_frames, height, width), device=pipe.device, dtype=pipe.torch_dtype)
        bbox_mask = pipe.preprocess_video(bbox_mask)
        bbox_latents = pipe.vae.encode(bbox_mask, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

        vid_f = num_frames
        lat_c, lat_f, lat_h, lat_w = bbox_latents.shape[1], bbox_latents.shape[2], bbox_latents.shape[3], bbox_latents.shape[4]

        if reference_imgs_indicator is None:
            reference_imgs_indicator = [1]

        has_reference_imgs = reference_imgs is not None and len(reference_imgs) > 0
        if has_reference_imgs:
            ref_tensor_list = []
            for img in reference_imgs:
                t = TF.to_tensor(img)
                t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ref_tensor_list.append(t)
            reference_imgs_tensor = torch.stack(ref_tensor_list).to(device=pipe.device, dtype=pipe.torch_dtype)
            reference_imgs_tensor = reference_imgs_tensor.unsqueeze(2)
            reference_imgs_tensor = pipe.preprocess_video(reference_imgs_tensor)
            reference_imgs_latents = pipe.vae.encode(
                reference_imgs_tensor, device=pipe.device, tiled=tiled,
                tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            obj_num = reference_imgs_latents.shape[0]
        elif object_bbox_masks is not None:
            obj_num = object_bbox_masks.shape[0]
        elif input_tracks is not None:
            obj_num = input_tracks.shape[2]
        else:
            obj_num = sum(reference_imgs_indicator)

        if pipe.use_special_embedding:
            group_ids = []
            for n in reference_imgs_indicator:
                ids = list(range(1, n + 1))
                group_ids.extend(ids)
            embed_weight_device = pipe.dit.group_id_embed.weight.device
            group_ids = torch.tensor(group_ids, dtype=torch.long, device=embed_weight_device)
        else:
            group_ids = torch.tensor([], dtype=torch.long, device=pipe.device)

        if pipe.use_special_embedding and len(group_ids) > 0:
            group_embeds_raw = pipe.dit.group_id_embed(group_ids)
            group_embeds = group_embeds_raw.to(device=pipe.device, dtype=pipe.torch_dtype).view(-1, lat_c, 1, 1, 1)
        else:
            group_embeds = torch.zeros(obj_num, lat_c, 1, 1, 1, device=pipe.device, dtype=pipe.torch_dtype)

        object_embeds = pipe.dit.object_embed.to(device=pipe.device, dtype=pipe.torch_dtype).view(1, lat_c, 1, 1, 1)
        control_emb = pipe.dit.control_signal_embed.to(device=pipe.device, dtype=pipe.torch_dtype).view(1, lat_c, 1, 1, 1)

        if has_reference_imgs:
            reference_imgs_latents = reference_imgs_latents + object_embeds + group_embeds
            ref_imgs_latents_per_sample = torch.split(reference_imgs_latents, reference_imgs_indicator, dim=0)
        else:
            ref_imgs_latents_per_sample = []

                                                                          
                              
        if object_bbox_masks is not None:
            object_bbox_masks = object_bbox_masks.float()
                                            
            lat_obj_bbox_masks = F.interpolate(
                object_bbox_masks.flatten(0, 1),                             
                size=(lat_h, lat_w),
                mode='nearest'
            ).view(obj_num, vid_f, 1, lat_h, lat_w)                             

                                      
            first_frame = lat_obj_bbox_masks[:, 0:1]                            
            avg_frames = []
            downsample_ratio_f = pipe.downsample_ratios[0]                  
            for start_idx in range(1, vid_f, downsample_ratio_f):  
                end_idx = min(start_idx + downsample_ratio_f, vid_f)
                frame_group = lat_obj_bbox_masks[:, start_idx:end_idx]                            
                avg_frame = frame_group.mean(dim=1, keepdim=True)                            
                avg_frames.append(avg_frame)
            lat_obj_bbox_masks = torch.cat([first_frame] + avg_frames, dim=1)                             

                          
            bbox_group_embeds = group_embeds * lat_obj_bbox_masks.permute(0, 2, 1, 3, 4)                              
            
                                     
            aggregated_embeds = []
            start_idx = 0
            for count in reference_imgs_indicator:
                sample_embeds = bbox_group_embeds[start_idx:start_idx+count]
                summed_embeds = sample_embeds.sum(dim=0)                                                 
                aggregated_embeds.append(summed_embeds)
                start_idx += count      
            bbox_group_embeds_final = torch.stack(aggregated_embeds).to(pipe.torch_dtype)                                 
            
                                           
            bbox_latents = bbox_latents + bbox_group_embeds_final + control_emb
            bbox_latents = bbox_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
                                                                                                                
                                           
                                            
                                                                                                           
        else:
            lat_obj_bbox_masks = None
            bbox_latents = bbox_latents + control_emb

                                                                          
                       
        track_info = None
        use_track_cond = True
        if input_tracks is not None and input_visibilities is not None:
            group_embeds_per_sample = torch.split(group_embeds.view(-1, lat_c), reference_imgs_indicator, dim=0)
            
                                
            def assign_obj_indices(obj_masks_sample):
                num_objs = obj_masks_sample.shape[0]
                assignment = torch.full((height, width), -1, dtype=torch.long, device=pipe.device)
                for obj_idx in range(num_objs):
                    mask = obj_masks_sample[obj_idx, 0, 0] > 0.5
                    assignment[mask] = obj_idx
                return assignment

            obj_assignments_list = []
            if object_masks is not None:
                for i in range(object_masks.shape[0]):
                    assignment = assign_obj_indices(object_masks[i])
                    obj_assignments_list.append(assignment)
                obj_assignments_batch = torch.stack(obj_assignments_list, dim=0)
            else:
                bsz = input_tracks.shape[0]
                n_tracks = input_tracks.shape[2]
                obj_assignments_batch = torch.full(
                    (bsz, height, width), -1, dtype=torch.long, device=pipe.device
                )
                obj_assignments_batch = obj_assignments_batch.unsqueeze(0).expand(bsz, -1, -1)
                for b in range(bsz):
                    for n in range(n_tracks):
                        pt = input_tracks[b, 0, n]
                        x, y = int(pt[0].item()), int(pt[1].item())
                        if 0 <= x < width and 0 <= y < height:
                            obj_assignments_batch[b, y, x] = n

            pred_tracks = input_tracks.to(pipe.device)
            pred_visibility = input_visibilities.to(pipe.device)

            track_video, track_pos = create_pos_feature_map(
                pred_tracks=pred_tracks,                
                pred_visibility=pred_visibility,             
                downsample_ratios=pipe.downsample_ratios,
                height=height,
                width=width,
                pos_emb_dim=lat_c, 
                control_signal_embed=pipe.dit.control_signal_embed, 
                group_embeds_per_sample=group_embeds_per_sample,
                obj_assignments_batch=obj_assignments_batch,
                track_channel_obj_slot=track_channel_obj_slot,
                track_num=-1,
                t_down_strategy="sample",
                device=pipe.device
            )
            
            track_video = track_video.permute(0, 4, 1, 2, 3)                                               
            track_video = track_video.to(dtype=pipe.torch_dtype)
            
        else:
            use_track_cond = False
            track_video = torch.zeros_like(noise).to(pipe.torch_dtype)

        return {
            "latents": noise,                        
            "ref_imgs_latents_per_sample": ref_imgs_latents_per_sample,                         
            "bbox_latents": bbox_latents,                        
            "track_video": track_video,                        
            "track_info": track_info, 
            "use_bbox_cond": use_bbox_cond, 
            "use_track_cond": use_track_cond, 
            "lat_obj_bbox_masks": lat_obj_bbox_masks                      
        }

class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}

class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        noise = pipe.generate_noise((1, 16, length, height//8, width//8), seed=seed, rand_device=rand_device)
        
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}

class WanVideoUnit_ImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega

class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states

class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype)                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value

def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
                   
    bbox_latents: torch.Tensor = None,
    track_video: torch.Tensor = None,
    ref_imgs_latents_per_sample: Optional[torch.Tensor] = None,
                                                            
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

                 
    x = latents + dit.bbox_zeroconv(bbox_latents)
                
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)                           
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    B, C, num_F, H, W = x.shape
    max_obj_num = 10
    num_objs = []
    has_ref_latents = ref_imgs_latents_per_sample is not None and len(ref_imgs_latents_per_sample) > 0
    if has_ref_latents:
        num_objs = [ref.shape[0] for ref in ref_imgs_latents_per_sample]
                      
        padded_refs = []
                            
                                                    
                                                                                            
        for i, ref_imgs in enumerate(ref_imgs_latents_per_sample):
            obj_num = min(num_objs[i], max_obj_num)
            padded = torch.zeros((max_obj_num, C, 1, H, W), device=x.device, dtype=x.dtype)
            if obj_num > 0:
                padded[:obj_num] = ref_imgs[:obj_num]
            padded_refs.append(padded)

        reference_imgs_latents = torch.stack(padded_refs, dim=0)
        reference_imgs_latents = reference_imgs_latents.squeeze(3).permute(0, 2, 1, 3, 4)
        x = torch.cat([x, reference_imgs_latents], dim=2)

    use_special_rope = kwargs.get("use_special_rope", True)
    F_total = x.shape[2]
    if use_special_rope:
        time_indices = torch.zeros((B, F_total), dtype=torch.long)
        for i in range(B):
            time_indices[i, :num_F] = torch.arange(num_F)
            if has_ref_latents:
                obj_num = min(num_objs[i], max_obj_num)
                time_indices[i, num_F:num_F + obj_num] = dit.ref_img_time_pos
                time_indices[i, num_F + obj_num:num_F + max_obj_num] = dit.invalid_ref_img_time_pos
    else:
        time_indices = torch.zeros((B, F_total), dtype=torch.long)
        for i in range(B):
            time_indices[i, :] = torch.arange(F_total)

                        
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    if hasattr(dit, 'bbox_zeroconv_layers'):
        bbox_latents_patchified, (_, _, _) = dit.patchify(bbox_latents)

                     
                                       
                                               
                                                            
                                                                                        
                                                         
                
    
                         
                                                                 
                                                                 
                                                                
                                                       

    if use_special_rope:
                
        h_idx = torch.arange(h)
        w_idx = torch.arange(w)
                                        
        time_indices_expanded = time_indices.view(B, F_total, 1, 1).expand(-1, -1, h, w)
                                                  
        time_freqs = dit.freqs[0][time_indices_expanded]
                                                  
        h_freqs = dit.freqs[1][h_idx].view(1, 1, h, 1, -1).expand(B, F_total, -1, w, -1)
                                                  
        w_freqs = dit.freqs[2][w_idx].view(1, 1, 1, w, -1).expand(B, F_total, h, -1, -1)
                  
        freqs = torch.cat([time_freqs, h_freqs, w_freqs], dim=-1)
                                      
        freqs = freqs.reshape(B, F_total * h * w, 1, -1).to(device=x.device)

        original_F = num_F
        if track_video is not None:
            track_video, (original_F, _, _) = dit.patchify(track_video)
                                 
            traj_time_indices = torch.zeros((B, original_F), dtype=torch.long)
            for i in range(B):
                traj_time_indices[i] = torch.arange(original_F)
                                               
            traj_time_indices_expanded = traj_time_indices.view(B, original_F, 1, 1).expand(-1, -1, h, w)
                        
            traj_time_freqs = dit.freqs[0][traj_time_indices_expanded]
            traj_h_freqs = dit.freqs[1][h_idx].view(1, 1, h, 1, -1).expand(B, original_F, -1, w, -1)
            traj_w_freqs = dit.freqs[2][w_idx].view(1, 1, 1, w, -1).expand(B, original_F, h, -1, -1)
            
            traj_freqs = torch.cat([traj_time_freqs, traj_h_freqs, traj_w_freqs], dim=-1)
            traj_freqs = traj_freqs.reshape(B, original_F * h * w, 1, -1).to(device=x.device)
            
            freqs = torch.cat([freqs, traj_freqs], dim=1)
            x = torch.cat([x, track_video], dim=1)
    else:
        track_video, (original_F, _, _) = dit.patchify(track_video)
        x = torch.cat([x, track_video], dim=1)
                
        h_idx = torch.arange(h)
        w_idx = torch.arange(w)
                                        
        time_indices_expanded = time_indices.view(B, F_total, 1, 1).expand(-1, -1, h, w)
                                                  
        time_freqs = dit.freqs[0][time_indices_expanded]
                                                  
        h_freqs = dit.freqs[1][h_idx].view(1, 1, h, 1, -1).expand(B, F_total, -1, w, -1)
                                                  
        w_freqs = dit.freqs[2][w_idx].view(1, 1, 1, w, -1).expand(B, F_total, h, -1, -1)
                  
        freqs = torch.cat([time_freqs, h_freqs, w_freqs], dim=-1)
                                      
        freqs = freqs.reshape(B, F_total * h * w, 1, -1).to(device=x.device)
    
              
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
            
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if hasattr(dit, 'bbox_zeroconv_layers'):
                current_layer = dit.bbox_zeroconv_layers[block_id].to(bbox_latents_patchified.device)
                bbox_latents_cur_layer = current_layer(bbox_latents_patchified.permute(0, 2, 1)).permute(0, 2, 1)
                vid_latent_seq = num_F*h*w
                x[:, :vid_latent_seq] += bbox_latents_cur_layer
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
                              
                                       
                                               
                
                                      

                             
    x = x[:, :original_F*h*w, :]                       
                        
    x = dit.unpatchify(x, (original_F, h, w))
    return x
