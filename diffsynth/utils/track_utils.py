from typing import Optional, Sequence

import torch
import numpy as np

SKIP_ZERO = True

def get_pos_emb(
    pos_k: torch.Tensor,
    pos_emb_dim: int,
    theta_func: callable = lambda i, d: torch.pow(10000, torch.mul(2, torch.div(i.to(torch.float32), d))),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
           
    assert pos_emb_dim % 2 == 0, "The dimension of position embeddings must be even."
    pos_k = pos_k.to(device=device, dtype=dtype)
    if SKIP_ZERO:
        pos_k = pos_k + 1
    batch_size = pos_k.size(0)

    denominator = torch.arange(0, pos_emb_dim // 2, device=device, dtype=dtype)
                                                                   
    denominator_expanded = denominator.view(1, -1).expand(batch_size, -1)
    
    thetas = theta_func(denominator_expanded, pos_emb_dim)
    
                                                           
    pos_k_expanded = pos_k.view(-1, 1).to(dtype)
    sin_thetas = torch.sin(torch.div(pos_k_expanded, thetas))
    cos_thetas = torch.cos(torch.div(pos_k_expanded, thetas))

                                                                     
    pos_emb = torch.cat([sin_thetas, cos_thetas], dim=-1)

    return pos_emb

def create_pos_feature_map(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    downsample_ratios: list[int],
    height: int,
    width: int,
    pos_emb_dim: int,
    control_signal_embed: torch.Tensor,
    group_embeds_per_sample: list[torch.Tensor],
    obj_assignments_batch: torch.Tensor,
    track_channel_obj_slot: Optional[Sequence[int]] = None,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    assert t_down_strategy in ["sample", "average"], "Invalid time downsampling strategy"
    
    B, T, N, _ = pred_tracks.shape
    t_down, h_down, w_down = downsample_ratios
    T_prime = (T - 1) // t_down + 1
    H_prime = height // h_down
    W_prime = width // w_down
    
    feature_map = torch.zeros(B, T_prime, H_prime, W_prime, pos_emb_dim, 
                             device=device, dtype=dtype)
    
    if track_num == -1:
        track_num = N
    
                    
    tracks_idx = torch.stack([
        torch.randperm(N, device=device)[:track_num] 
        for _ in range(B)
    ])
    
                             
    global_embs = get_pos_emb(torch.arange(N, device=device), pos_emb_dim, device=device, dtype=dtype)
    tracks_embs = global_embs[tracks_idx]                               
    
                                      
    for i in range(B):
        group_embeds = group_embeds_per_sample[i]                             
        num_objs = group_embeds.shape[0]
        
                                                         
        sample_tracks_idx = tracks_idx[i]               
        initial_positions = pred_tracks[i, 0, sample_tracks_idx]                  
        
                                  
        if track_channel_obj_slot is not None:
            slot_map = torch.tensor(track_channel_obj_slot, device=device, dtype=torch.long)
            obj_indices = slot_map[sample_tracks_idx]
        else:
            x_indices = initial_positions[:, 0].clamp(0, width - 1).round().long()
            y_indices = initial_positions[:, 1].clamp(0, height - 1).round().long()
            obj_indices = obj_assignments_batch[i, y_indices, x_indices]

        valid_mask = (obj_indices >= 0) & (obj_indices < num_objs)
        
                                                    
        if valid_mask.any():
            tracks_embs[i, valid_mask] += group_embeds[obj_indices[valid_mask]]
    
                                  
    if control_signal_embed is not None:
        tracks_embs += control_signal_embed.reshape(1, 1, -1).to(tracks_embs.device)
    
                                          
    t_indices = torch.arange(0, T, t_down, device=device)[:T_prime]
    if t_down_strategy == "sample":
                                                     
        sampled_tracks = pred_tracks[:, t_indices]
        sampled_visibility = pred_visibility[:, t_indices]
    else:             
                                          
        time_windows = []
        for t_idx in t_indices:
            end = min(t_idx + t_down, T)
            if t_idx == 0:                                  
                window = slice(t_idx, t_idx + 1)
            else:
                window = slice(t_idx, end)
            time_windows.append(window)
        
                                  
        sampled_tracks = []
        sampled_visibility = []
        
        for window in time_windows:
                                                          
            vis_window = torch.any(
                pred_visibility[:, window], 
                dim=1,
                keepdim=True
            )             
            
                                                        
            pos_window = pred_tracks[:, window]                
            
                                                           
            valid_mask = pred_visibility[:, window].unsqueeze(-1)                
            pos_sum = (pos_window * valid_mask).sum(dim=1)             
            valid_count = valid_mask.sum(dim=1).clamp(min=1)             
            avg_pos = pos_sum / valid_count
            
            sampled_tracks.append(avg_pos.unsqueeze(1))
            sampled_visibility.append(vis_window.unsqueeze(1))
        
        sampled_tracks = torch.cat(sampled_tracks, dim=1)                          
        sampled_visibility = torch.cat(sampled_visibility, dim=1)                   
    
                                                      
    batch_idx = torch.arange(B, device=device)[:, None, None]             
    time_idx = torch.arange(T_prime, device=device)[None, :, None]                   
    sampled_tracks = sampled_tracks[batch_idx, time_idx, tracks_idx.unsqueeze(1)]
    sampled_visibility = sampled_visibility[batch_idx, time_idx, tracks_idx.unsqueeze(1)]

                                              
                                                                      
    x_coords = (sampled_tracks[..., 0] / w_down).long()                       
    y_coords = (sampled_tracks[..., 1] / h_down).long()                       
    
                                               
    valid_mask = sampled_visibility &                 (x_coords >= 0) & (x_coords < W_prime) &                 (y_coords >= 0) & (y_coords < H_prime)
    
                                    
                                 
    b_idx, t_idx, tr_idx = torch.where(valid_mask)
    
                                                  
    x_valid = x_coords[b_idx, t_idx, tr_idx]
    y_valid = y_coords[b_idx, t_idx, tr_idx]
    embeddings_valid = tracks_embs[b_idx, tr_idx]                          
    
                                
    feature_map.index_put_(
        indices=(b_idx, t_idx, y_valid, x_valid),
        values=embeddings_valid,
        accumulate=True
    )
    
    return feature_map, None

