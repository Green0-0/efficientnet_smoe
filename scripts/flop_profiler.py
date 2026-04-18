"""
flop_profiler.py

A robust, hook-based MAC/FLOP profiler designed specifically for EfficientNet 
and MoE-modified (DeepMoE) variants. It dynamically separates static overhead 
from dynamic, per-channel routing costs.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

# ==========================================
# 1. MATHEMATICAL HELPERS
# ==========================================

def _calculate_conv2d_macs(module: nn.Conv2d, output_shape: torch.Size) -> float:
    h_out, w_out = output_shape[2], output_shape[3]
    k_h, k_w = module.kernel_size
    c_in = module.in_channels
    c_out = module.out_channels
    groups = module.groups
    return float(h_out * w_out * (c_in // groups) * c_out * k_h * k_w)

def _calculate_linear_macs(module: nn.Linear) -> float:
    return float(module.in_features * module.out_features)

def _calculate_pool2d_macs(input_shape: torch.Size) -> float:
    h_in, w_in = input_shape[2], input_shape[3]
    c_in = input_shape[1]
    return float(h_in * w_in * c_in)

# ==========================================
# 2. TOPOLOGY PARSING & TAGGING
# ==========================================

def _tag_network_topology(model: nn.Module) -> Dict[int, Dict[str, Any]]:
    tags = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d)):
            tags[id(module)] = {'role': 'static'}

    features = None
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'features'):
        features = model.base_model.features
    elif hasattr(model, 'features'):
        features = model.features

    if features is None:
        return tags 

    block_idx = 0
    for stage_idx in range(1, 8):
        if stage_idx >= len(features):
            break
            
        for block in features[stage_idx]:
            b_id = str(block_idx)
            is_gated = hasattr(model, 'gates') and b_id in model.gates
            seen_depthwise = False
            conv1x1_after_dw = 0 
            
            for name, m in block.named_modules():
                if isinstance(m, nn.Conv2d):
                    if m.groups == m.in_channels and m.in_channels > 1:
                        tags[id(m)] = {
                            'role': 'dynamic_depthwise', 
                            'block_idx': b_id, 
                            'is_gated': is_gated
                        }
                        seen_depthwise = True
                        conv1x1_after_dw = 0 
                        
                    elif m.kernel_size == (1, 1) or m.kernel_size == 1:
                        if seen_depthwise:
                            conv1x1_after_dw += 1
                            if conv1x1_after_dw == 1:
                                tags[id(m)] = {'role': 'dynamic_se_fc1', 'block_idx': b_id}
                            elif conv1x1_after_dw == 2:
                                tags[id(m)] = {'role': 'dynamic_se_fc2', 'block_idx': b_id}
                            else:
                                tags[id(m)] = {'role': 'dynamic_project', 'block_idx': b_id}
                        else:
                            # 1x1 Conv prior to depthwise is the Expand layer
                            tags[id(m)] = {'role': 'dynamic_expand', 'block_idx': b_id}
                            
                elif isinstance(m, nn.AdaptiveAvgPool2d):
                    if seen_depthwise: # Only tag the SE pool, ignore the classification head pool
                        tags[id(m)] = {'role': 'dynamic_se_pool', 'block_idx': b_id}
                    
            block_idx += 1

    return tags

# ==========================================
# 3. MAIN PROFILER API
# ==========================================

def profile_deepmoe_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_size, device=device)
    
    tags = _tag_network_topology(model)
    
    profiler_state = {
        'static_flops': 0.0,
        'total_dynamic_flops': 0.0,
        'flops_per_channel': {}
    }
    
    handles = []

    def compute_macs_hook(module, inputs, output):
        x = inputs[0]
        macs = 0.0
        c_in = float(x.shape[1])
        c_out = float(output.shape[1])
        
        if isinstance(module, nn.Conv2d):
            macs = _calculate_conv2d_macs(module, output.shape)
        elif isinstance(module, nn.Linear):
            macs = _calculate_linear_macs(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            macs = _calculate_pool2d_macs(x.shape)
            
        tag_info = tags.get(id(module), {'role': 'ignore'})
        role = tag_info.get('role')
        
        if role == 'static':
            profiler_state['static_flops'] += macs
            
        elif role in ['dynamic_depthwise', 'dynamic_project', 'dynamic_se_pool', 'dynamic_se_fc1']:
            b_idx = tag_info['block_idx']
            profiler_state['total_dynamic_flops'] += macs
            
            # Gated dimension is the INPUT
            cost_per_channel = macs / max(1.0, c_in)
            profiler_state['flops_per_channel'][b_idx] = profiler_state['flops_per_channel'].get(b_idx, 0.0) + cost_per_channel
            
            # Gate application phantom FLOPs (Moved to per-channel dynamic)
            if role == 'dynamic_depthwise' and tag_info.get('is_gated', False):
                gate_cost_per_channel = float(output.shape[2] * output.shape[3])
                profiler_state['flops_per_channel'][b_idx] += gate_cost_per_channel
                profiler_state['total_dynamic_flops'] += (gate_cost_per_channel * c_in)

        elif role in ['dynamic_se_fc2', 'dynamic_expand']:
            b_idx = tag_info['block_idx']
            profiler_state['total_dynamic_flops'] += macs
            
            # Gated dimension is the OUTPUT
            cost_per_channel = macs / max(1.0, c_out)
            profiler_state['flops_per_channel'][b_idx] = profiler_state['flops_per_channel'].get(b_idx, 0.0) + cost_per_channel

    for module in model.modules():
        if id(module) in tags:
            handles.append(module.register_forward_hook(compute_macs_hook))

    is_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    if is_training:
        model.train()

    for handle in handles:
        handle.remove()

    total_baseline_flops = profiler_state['static_flops'] + profiler_state['total_dynamic_flops']

    return {
        'static': profiler_state['static_flops'],
        'per_channel': profiler_state['flops_per_channel'],
        'total': total_baseline_flops
    }