import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def warmup_quant(model, loader, warmup_batches=10):
    model.train()
    print(f"[Warmup] Running {warmup_batches} warmup batches to calibrate quantization ranges...")

    for i, batch_x in enumerate(loader):  # 只取一個元素
        if i >= warmup_batches:
            break
        if isinstance(batch_x, (list, tuple)):
            batch_x = batch_x[0]
        batch_x = batch_x.to(next(model.parameters()).device)
        _ = model(batch_x)
    

# ---------------------------
# Per-tensor POT quantizer (STE)
# ---------------------------
class QuantizePOTModule(nn.Module):
    """
    Power-of-two per-tensor quantizer with STE.
    signed=True : scale = 2^{-(b-1)}, q in [-2^{b-1}, 2^{b-1}-1], zero_point=0
    signed=False: scale = 2^{-b},     q in [0, 2^b-1],             zero_point=0
    """
    def __init__(self, num_bits: int = 8, signed: bool = True, eps: float = 1e-6):
        super().__init__()
        assert num_bits >= 2 if signed else num_bits >= 1
        self.num_bits = num_bits
        self.signed = signed
        self.eps = eps

        if signed:
            self.qmin = -(1 << (num_bits - 1))
            self.qmax =  (1 << (num_bits - 1)) - 1
            scale = 2.0 ** (-(num_bits - 1))
        else:
            self.qmin = 0
            self.qmax = (1 << num_bits) - 1
            scale = 2.0 ** (-num_bits)

        # expose as buffers so hooks/export can see them
        self.register_buffer("scale_buf", torch.tensor(float(scale)), persistent=False)
        self.register_buffer("zero_point_buf", torch.tensor(0.0), persistent=False)


        self.register_buffer("q_int_buf", torch.tensor([], dtype=torch.int32), persistent=False) 
        self.register_buffer("dequant_buf", torch.tensor([], dtype=torch.float32), persistent=False)  


    @property
    def scale(self):
        # for hooks: exposes a tensor
        return self.scale_buf

    @property
    def zero_point(self):
        # always 0 for this POT scheme
        return self.zero_point_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # robust clamp to the final representable bin
        if self.signed:
            x = x.clamp(-1.0, 1.0 - self.scale_buf + self.eps)
        else:
            x = x.clamp(0.0, 1.0 - self.scale_buf + self.eps)

        q = torch.trunc(x / self.scale_buf)
        q = torch.clamp(q, self.qmin, self.qmax)
        x_q = q * self.scale_buf

        with torch.no_grad():
            self.q_int_buf.resize_as_(q).copy_(q.to(torch.int32))
            self.dequant_buf.resize_as_(x_q).copy_(x_q)

        # STE
        return x + (x_q - x).detach()


# ---------------------------
# Per-channel POT quantizer for weights (STE)
# ch_axis=0 for Conv/Linear out_channels/out_features
# ---------------------------
class PerChannelPOTQuant(nn.Module):
    """
    Per-channel POT quantizer (shift-only) with STE.
    Typically for weights: signed=True, symmetric, ch_axis=0.
    Exposes latest per-channel scale and zero_point (0) after forward.
    """
    def __init__(self, num_bits: int = 8, signed: bool = True, ch_axis: int = 0, eps: float = 1e-12):
        super().__init__()
        assert num_bits >= 2 if signed else num_bits >= 1
        self.num_bits = num_bits
        self.signed = signed
        self.ch_axis = ch_axis
        self.eps = eps

        if signed:
            self.qmin = -(1 << (num_bits - 1))
            self.qmax =  (1 << (num_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << num_bits) - 1

        # frozen k (log2(scale)) if you choose to freeze scales
        self.register_buffer("frozen_k", None)  # tensor [C] or None

        # latest used (exposed for hooks); set after forward
        self.latest_scale = None        # tensor [C]
        self.latest_zero_point = None   # tensor [C] (zeros)

        # buffers for dumping
        self.register_buffer("q_int_buf", torch.tensor([], dtype=torch.int32), persistent=False)
        self.register_buffer("dequant_buf", torch.tensor([], dtype=torch.float32), persistent=False)

    def _reshape_scale(self, s: torch.Tensor, x: torch.Tensor):
        shape = [1] * x.dim()
        shape[self.ch_axis] = -1
        return s.view(*shape)

    @torch.no_grad()
    def freeze_scales(self, x_sample: torch.Tensor):
        """Estimate & freeze per-channel scales (k=round(log2(s))) from a sample."""
        reduce_dims = [d for d in range(x_sample.dim()) if d != self.ch_axis]
        if self.signed:
            max_abs = x_sample.abs().amax(dim=reduce_dims, keepdim=False)
        else:
            max_abs = x_sample.amax(dim=reduce_dims, keepdim=False)
        max_abs = torch.clamp(max_abs, min=self.eps)
        s_real = max_abs / float(self.qmax)    # [C]
        k = torch.round(torch.log2(s_real))    # [C]
        self.frozen_k = k

    @property
    def scale(self):
        # prefer last used scale; otherwise derive from frozen_k if present
        if self.latest_scale is not None:
            return self.latest_scale
        if self.frozen_k is not None:
            base = self.frozen_k.new_tensor(2.0)
            return torch.pow(base, self.frozen_k)
        return None

    @property
    def zero_point(self):
        if self.latest_zero_point is not None:
            return self.latest_zero_point
        if self.frozen_k is not None:
            return torch.zeros_like(self.frozen_k)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduce_dims = [d for d in range(x.dim()) if d != self.ch_axis]

        if self.frozen_k is None:
            if self.signed:
                max_abs = x.abs().amax(dim=reduce_dims, keepdim=False)  # [C]
            else:
                max_abs = x.amax(dim=reduce_dims, keepdim=False)        # [C]
            max_abs = torch.clamp(max_abs, min=self.eps)
            s_real = max_abs / float(self.qmax)                         # [C]
            k = torch.round(torch.log2(s_real))                         # [C]
        else:
            k = self.frozen_k

        # compute per-channel scale (POT) used this forward
        base = x.new_tensor(2.0)
        s_pot = torch.pow(base, k)                 # [C]
        s_b   = self._reshape_scale(s_pot, x)      # broadcast to x

        # expose for hooks
        self.latest_scale = s_pot.detach()
        self.latest_zero_point = torch.zeros_like(s_pot, device=s_pot.device, dtype=s_pot.dtype)

        # quantize with STE
        q   = torch.trunc(x / s_b)
        q   = torch.clamp(q, self.qmin, self.qmax)
        x_q = q * s_b

        # buffers for dump
        with torch.no_grad():
            self.q_int_buf.resize_as_(q).copy_(q.to(torch.int32))
            self.dequant_buf.resize_as_(x_q).copy_(x_q.to(dtype=x.dtype))

        # STE
        return x + (x_q - x).detach()


# ------------------------------------------------
# Quantized Conv1d / ConvTranspose1d
# Expose per-layer quant params so hooks can persist them.
# ------------------------------------------------
class QuantConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 act_bits=8, weight_bits=8, out_bits=8, act_signed=True, weight_signed=True,
                 quantize_activation=False, freeze_weight_scales=False):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        out_signed = (act_signed or weight_signed)

        # activation quant (per-tensor POT; optional)
        self.quantize_activation = quantize_activation
        self.act_quant = QuantizePOTModule(act_bits, signed=act_signed) if quantize_activation else None

        # weight quant (per-channel POT, along out_channels axis=0)
        self.weight_quant = PerChannelPOTQuant(weight_bits, signed=weight_signed, ch_axis=0)

        # bias quant (per-channel POT, along out_channels axis=0)
        self.bias_quant = PerChannelPOTQuant(32, signed=out_signed, ch_axis=0)

        # output quant (per-tensor POT)
        self.out_quant = QuantizePOTModule(out_bits, signed=out_signed)

        # optionally freeze weight scales
        self.freeze_weight_scales_flag = freeze_weight_scales
        if self.freeze_weight_scales_flag:
            self.weight_quant.freeze_scales(self.weight.detach())

        # 執行期觀察 buffer（不進 state_dict）
        self.register_buffer("weight_int", torch.zeros_like(self.weight, dtype=torch.int32),   persistent=False)
        self.register_buffer("weight_dq",  torch.zeros_like(self.weight, dtype=torch.float32), persistent=False)
        self.register_buffer("bias_int", torch.tensor([], dtype=torch.int32),   persistent=False)
        self.register_buffer("bias_dq",  torch.tensor([], dtype=torch.float32), persistent=False)
        self.register_buffer("out_int",    torch.tensor([], dtype=torch.int32),   persistent=False)
        self.register_buffer("out_dq",     torch.tensor([], dtype=torch.float32), persistent=False)

        # placeholders exposed for hooks after forward
        self.scale = self.zero_point = None
        self.w_scale = self.w_zero_point = None
        self.b_scale = self.b_zero_point = None
        self.a_scale = self.a_zero_point = None
        self.out_scale = self.out_zero_point = None

    def forward(self, x):
        # activation
        if self.act_quant is not None:
            x_q = self.act_quant(x)
            self.a_scale = self.act_quant.scale
            self.a_zero_point = self.act_quant.zero_point
        else:
            x_q = x
            self.a_scale = self.a_zero_point = None

        # weight
        w_q = self.weight_quant(self.weight)
        self.w_scale = self.weight_quant.scale
        self.w_zero_point = self.weight_quant.zero_point
        with torch.no_grad():
            if self.weight_quant.q_int_buf.numel() > 0:
                self.weight_int.resize_as_(self.weight_quant.q_int_buf).copy_(self.weight_quant.q_int_buf)
            if self.weight_quant.dequant_buf.numel() > 0:
                self.weight_dq.resize_as_(self.weight_quant.dequant_buf).copy_(self.weight_quant.dequant_buf)

        # bias
        b_q = None
        if self.bias is not None:
            S_in = (self.act_quant.scale if self.act_quant is not None else x.new_tensor(1.0))
            S_w = self.weight_quant.scale.to(x.device)   # [C_out]
            S_b = S_w * S_in

            with torch.no_grad():
                k = torch.round(torch.log2(S_b))
                self.bias_quant.frozen_k = k

            b_q = self.bias_quant(self.bias)
            self.b_scale = self.bias_quant.scale         # -> [C_out]
            self.b_zero_point = self.bias_quant.zero_point 
            with torch.no_grad():
                if getattr(self.bias_quant, "q_int_buf", None) is not None and self.bias_quant.q_int_buf.numel() > 0:
                    self.bias_int.resize_as_(self.bias_quant.q_int_buf).copy_(self.bias_quant.q_int_buf)

                if getattr(self.bias_quant, "dequant_buf", None) is not None and self.bias_quant.dequant_buf.numel() > 0:
                    self.bias_dq.resize_as_(self.bias_quant.dequant_buf).copy_(self.bias_quant.dequant_buf)


        # conv
        out = F.conv1d(x_q, w_q, b_q, self.stride, self.padding, self.dilation, self.groups)

        # output
        out_q = self.out_quant(out)
        self.out_scale = self.out_quant.scale
        self.out_zero_point = self.out_quant.zero_point
        with torch.no_grad():
            if self.out_quant.q_int_buf.numel() > 0:
                self.out_int.resize_as_(self.out_quant.q_int_buf).copy_(self.out_quant.q_int_buf)
            if self.out_quant.dequant_buf.numel() > 0:
                self.out_dq.resize_as_(self.out_quant.dequant_buf).copy_(self.out_quant.dequant_buf)

        return out_q

class QuantConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 act_bits=8, weight_bits=8, out_bits=8, act_signed=True, weight_signed=True,
                 quantize_activation=False, freeze_weight_scales=False):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

        out_signed = (act_signed or weight_signed)

        self.quantize_activation = quantize_activation
        self.act_quant  = QuantizePOTModule(act_bits,  signed=act_signed) if quantize_activation else None
        self.weight_quant = PerChannelPOTQuant(weight_bits, signed=weight_signed, ch_axis=1)
        self.bias_quant   = PerChannelPOTQuant(32,        signed=out_signed,    ch_axis=0)
        self.out_quant    = QuantizePOTModule(out_bits,   signed=out_signed)

        if freeze_weight_scales:
            self.weight_quant.freeze_scales(self.weight.detach())

        self.register_buffer("weight_int", torch.zeros_like(self.weight, dtype=torch.int32),   persistent=False)
        self.register_buffer("weight_dq",  torch.zeros_like(self.weight, dtype=torch.float32), persistent=False)
        self.register_buffer("bias_int", torch.tensor([], dtype=torch.int32),   persistent=False)
        self.register_buffer("bias_dq",  torch.tensor([], dtype=torch.float32), persistent=False)
        self.register_buffer("out_int",    torch.tensor([], dtype=torch.int32),   persistent=False)
        self.register_buffer("out_dq",     torch.tensor([], dtype=torch.float32), persistent=False)

        self.scale = self.zero_point = None
        self.w_scale = self.w_zero_point = None
        self.b_scale = self.b_zero_point = None
        self.a_scale = self.a_zero_point = None
        self.out_scale = self.out_zero_point = None

    def forward(self, x):
        # activation
        if self.act_quant is not None:
            x_q = self.act_quant(x)
            self.a_scale = self.act_quant.scale
            self.a_zero_point = self.act_quant.zero_point
        else:
            x_q = x
            self.a_scale = self.a_zero_point = None

        # weight
        w_q = self.weight_quant(self.weight)
        self.w_scale = self.weight_quant.scale
        self.w_zero_point = self.weight_quant.zero_point
        with torch.no_grad():
            if self.weight_quant.q_int_buf.numel() > 0:
                self.weight_int.resize_as_(self.weight_quant.q_int_buf).copy_(self.weight_quant.q_int_buf)
            if self.weight_quant.dequant_buf.numel() > 0:
                self.weight_dq.resize_as_(self.weight_quant.dequant_buf).copy_(self.weight_quant.dequant_buf)

        # bias
        b_q = None
        if self.bias is not None:
            S_in = (self.act_quant.scale if self.act_quant is not None else x.new_tensor(1.0))
            S_w = self.weight_quant.scale.to(x.device)   # [C_out]
            S_b = S_w * S_in

            with torch.no_grad():
                k = torch.round(torch.log2(S_b))
                self.bias_quant.frozen_k = k

            b_q = self.bias_quant(self.bias)
            self.b_scale = self.bias_quant.scale         # -> [C_out]
            self.b_zero_point = self.bias_quant.zero_point 

            with torch.no_grad():
                if self.bias_quant.q_int_buf.numel() > 0:
                    self.bias_int.resize_as_(self.bias_quant.q_int_buf).copy_(self.bias_quant.q_int_buf)
                if self.bias_quant.dequant_buf.numel() > 0:
                    self.bias_dq.resize_as_(self.bias_quant.dequant_buf).copy_(self.bias_quant.dequant_buf)

        # deconv
        out = F.conv_transpose1d(
            x_q, w_q, b_q, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation
        )

        # output
        out_q = self.out_quant(out)
        self.out_scale = self.out_quant.scale
        self.out_zero_point = self.out_quant.zero_point
        with torch.no_grad():
            if self.out_quant.q_int_buf.numel() > 0:
                self.out_int.resize_as_(self.out_quant.q_int_buf).copy_(self.out_quant.q_int_buf)
            if self.out_quant.dequant_buf.numel() > 0:
                self.out_dq.resize_as_(self.out_quant.dequant_buf).copy_(self.out_quant.dequant_buf)

        return out_q


if __name__ == "__main__":
    # simple test
    conv = QuantConv1d(3, 8, kernel_size=3, stride=1, padding=1,
                       act_bits=8, weight_bits=8, out_bits=8,
                       act_signed=True, weight_signed=True,
                       quantize_activation=True,
                       freeze_weight_scales=True)
    x = torch.randn(2, 3, 16)
    y = conv(x)
    print("Output shape:", y.shape)