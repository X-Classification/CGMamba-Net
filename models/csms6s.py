# --------------------------------------------------------
# Modified by Abdelrahman Shaker
# --------------------------------------------------------
# VMamba: Visual State Space Model
# Licensed under The MIT License [see LICENSE for details]
# Written by MzeroMiko
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
import math


# original scans =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb;
            pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# ============ Scans =============

# 1.  (Center to Edge - Clockwise)
class CrossScan_ConcentricOut_Clockwise(torch.autograd.Function):
    @staticmethod
    def get_concentric_clockwise_indices(H, W, device, outward=True):
        center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
        

        points = []
        for y in range(H):
            for x in range(W):
                dy, dx = y - center_y, x - center_x
                distance = math.sqrt(dy * dy + dx * dx)
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                points.append({'y': y, 'x': x, 'distance': distance, 'angle': angle, 'index': y * W + x})
        
        distance_groups = {}
        for point in points:
            dist_key = round(point['distance'] * 10000) / 10000
            if dist_key not in distance_groups:
                distance_groups[dist_key] = []
            distance_groups[dist_key].append(point)
        
        sorted_indices = []
        last_point = None
        
        if outward:
            distance_order = sorted(distance_groups.keys())
        else:
            distance_order = sorted(distance_groups.keys(), reverse=True)
        
        for dist_idx, dist in enumerate(distance_order):
            group = distance_groups[dist]
            
            if len(group) == 1:
                sorted_indices.append(group[0]['index'])
                last_point = group[0]
                continue

            if dist_idx == 0 or last_point is None:

                start_point = min(group, key=lambda p: (p['y'], p['x']))
            else:

                def distance_to_last_point(point):
                    dy = point['y'] - last_point['y']
                    dx = point['x'] - last_point['x']
                    return math.sqrt(dy * dy + dx * dx)
                start_point = min(group, key=distance_to_last_point)
            

            start_angle = start_point['angle']
            def clockwise_angle_from_start(point):
                angle_diff = point['angle'] - start_angle
                if angle_diff < 0:
                    angle_diff += 2 * math.pi
                return angle_diff
            
            group_sorted = sorted(group, key=clockwise_angle_from_start)
            for point in group_sorted:
                sorted_indices.append(point['index'])
            last_point = group_sorted[-1]
        
        return torch.tensor(sorted_indices, device=device, dtype=torch.long)

    @staticmethod
    def forward(ctx, x: torch.Tensor, outward=True):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        ctx.outward = outward
        device = x.device

        indices = CrossScan_ConcentricOut_Clockwise.get_concentric_clockwise_indices(
            H, W, device, outward=outward
        )
        

        xs = torch.gather(x.flatten(2, 3), dim=2, index=indices.expand(B, C, -1))
        ctx.save_for_backward(indices)
        return xs.unsqueeze(1)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        indices, = ctx.saved_tensors
        y_flat = ys.squeeze(1)
        y_unordered = torch.zeros(B, C, H * W, device=y_flat.device, dtype=y_flat.dtype)
        y_unordered.scatter_(2, indices.expand(B, C, -1), y_flat)
        return y_unordered.view(B, C, H, W), None


# 2.  (Center to Edge - Counter-Clockwise)
class CrossScan_ConcentricOut_CounterClockwise(torch.autograd.Function):
    @staticmethod
    def get_concentric_counter_clockwise_indices(H, W, device, outward=True):
        center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
        
        points = []
        for y in range(H):
            for x in range(W):
                dy, dx = y - center_y, x - center_x
                distance = math.sqrt(dy * dy + dx * dx)
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                points.append({'y': y, 'x': x, 'distance': distance, 'angle': angle, 'index': y * W + x})
        
        distance_groups = {}
        for point in points:
            dist_key = round(point['distance'] * 10000) / 10000
            if dist_key not in distance_groups:
                distance_groups[dist_key] = []
            distance_groups[dist_key].append(point)
        
        sorted_indices = []
        last_point = None
        
        if outward:
            distance_order = sorted(distance_groups.keys())
        else:
            distance_order = sorted(distance_groups.keys(), reverse=True)
        
        for dist_idx, dist in enumerate(distance_order):
            group = distance_groups[dist]
            
            if len(group) == 1:
                sorted_indices.append(group[0]['index'])
                last_point = group[0]
                continue
            

            if dist_idx == 0 or last_point is None:

                start_point = min(group, key=lambda p: (p['x'], p['y']))
            else:

                def distance_to_last_point(point):
                    dy = point['y'] - last_point['y']
                    dx = point['x'] - last_point['x']
                    return math.sqrt(dy * dy + dx * dx)
                start_point = min(group, key=distance_to_last_point)
            

            start_angle = start_point['angle']
            def counter_clockwise_angle_from_start(point):
                angle_diff = start_angle - point['angle']
                if angle_diff < 0:
                    angle_diff += 2 * math.pi
                return angle_diff
            
            group_sorted = sorted(group, key=counter_clockwise_angle_from_start)
            for point in group_sorted:
                sorted_indices.append(point['index'])
            last_point = group_sorted[-1]
        
        return torch.tensor(sorted_indices, device=device, dtype=torch.long)

    @staticmethod
    def forward(ctx, x: torch.Tensor, outward=True):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        ctx.outward = outward
        device = x.device


        indices = CrossScan_ConcentricOut_CounterClockwise.get_concentric_counter_clockwise_indices(
            H, W, device, outward=outward
        )
        
        xs = torch.gather(x.flatten(2, 3), dim=2, index=indices.expand(B, C, -1))
        ctx.save_for_backward(indices)
        return xs.unsqueeze(1)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        indices, = ctx.saved_tensors
        y_flat = ys.squeeze(1)
        y_unordered = torch.zeros(B, C, H * W, device=y_flat.device, dtype=y_flat.dtype)
        y_unordered.scatter_(2, indices.expand(B, C, -1), y_flat)
        return y_unordered.view(B, C, H, W), None



class CrossScan_Concentric_Clockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return CrossScan_ConcentricOut_Clockwise.forward(ctx, x, outward=True)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        return CrossScan_ConcentricOut_Clockwise.backward(ctx, ys)


class CrossScan_Convergence_Clockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return CrossScan_ConcentricOut_Clockwise.forward(ctx, x, outward=False)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        return CrossScan_ConcentricOut_Clockwise.backward(ctx, ys)


class CrossScan_Concentric_CounterClockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return CrossScan_ConcentricOut_CounterClockwise.forward(ctx, x, outward=True)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        return CrossScan_ConcentricOut_CounterClockwise.backward(ctx, ys)

class CrossScan_Convergence_CounterClockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return CrossScan_ConcentricOut_CounterClockwise.forward(ctx, x, outward=False)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        return CrossScan_ConcentricOut_CounterClockwise.backward(ctx, ys)


# Merge
class CrossMerge_Concentric_Clockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        device = ys.device


        indices = CrossScan_ConcentricOut_Clockwise.get_concentric_clockwise_indices(H, W, device, outward=True)

        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]  

        y_unordered = torch.zeros(B, D, H * W, device=y.device, dtype=y.dtype)
        y_unordered.scatter_(2, indices.expand(B, D, -1), y)
        return y_unordered

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        device = x.device

        indices = CrossScan_ConcentricOut_Clockwise.get_concentric_clockwise_indices(H, W, device, outward=True)

        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.gather(x, dim=2, index=indices.expand(B, C, -1))
        xs = xs.view(B, 1, C, H, W)
        return xs


class CrossMerge_Convergence_Clockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        device = ys.device

        indices = CrossScan_ConcentricOut_Clockwise.get_concentric_clockwise_indices(H, W, device, outward=False)

        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]

        y_unordered = torch.zeros(B, D, H * W, device=y.device, dtype=y.dtype)
        y_unordered.scatter_(2, indices.expand(B, D, -1), y)
        return y_unordered

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        device = x.device

        indices = CrossScan_ConcentricOut_Clockwise.get_concentric_clockwise_indices(H, W, device, outward=False)

        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.gather(x, dim=2, index=indices.expand(B, C, -1))
        xs = xs.view(B, 1, C, H, W)
        return xs


class CrossMerge_Concentric_CounterClockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        device = ys.device

        indices = CrossScan_ConcentricOut_CounterClockwise.get_concentric_counter_clockwise_indices(H, W, device, outward=True)

        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]

        y_unordered = torch.zeros(B, D, H * W, device=y.device, dtype=y.dtype)
        y_unordered.scatter_(2, indices.expand(B, D, -1), y)
        return y_unordered

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        device = x.device

        indices = CrossScan_ConcentricOut_CounterClockwise.get_concentric_counter_clockwise_indices(H, W, device, outward=True)

        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.gather(x, dim=2, index=indices.expand(B, C, -1))
        xs = xs.view(B, 1, C, H, W)
        return xs


class CrossMerge_Convergence_CounterClockwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        device = ys.device

        indices = CrossScan_ConcentricOut_CounterClockwise.get_concentric_counter_clockwise_indices(H, W, device, outward=False)

        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]

        y_unordered = torch.zeros(B, D, H * W, device=y.device, dtype=y.dtype)
        y_unordered.scatter_(2, indices.expand(B, D, -1), y)
        return y_unordered

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        device = x.device

        indices = CrossScan_ConcentricOut_CounterClockwise.get_concentric_counter_clockwise_indices(H, W, device, outward=False)

        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.gather(x, dim=2, index=indices.expand(B, C, -1))
        xs = xs.view(B, 1, C, H, W)
        return xs