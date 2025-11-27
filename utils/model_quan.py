import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

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
    
# class quantize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_: torch.Tensor, num_of_bits: int, signed: bool) -> torch.Tensor:
#         # Setup quant range
#         qmin = -(1 << (num_of_bits - 1)) if signed else 0
#         qmax = (1 << (num_of_bits - 1)) - 1 if signed else (1 << num_of_bits) - 1

#         # Compute scale and zero_point
#         x_min, x_max = input_.min(), input_.max()
#         scale = (x_max - x_min).clamp(min=1e-8) / (qmax - qmin)
#         zero_point = torch.round(qmin - x_min / scale).clamp(qmin, qmax)

#         # Quantize and dequantize
#         q_x = torch.round(input_ / scale + zero_point).clamp(qmin, qmax)
#         x = (q_x - zero_point) * scale
#         return x

#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor):
#         grad_input = grad_output.clone()
#         return grad_input, None, None
    
## POT 量化 + STE
class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, num_of_bits: int, signed: bool=True):

        if signed:
            qmin = -(1 << (num_of_bits - 1))
            qmax =  (1 << (num_of_bits - 1)) - 1
            scale = 2.0 ** (-(num_of_bits - 1))
            
        else:
            qmin = 0
            qmax = (1 << num_of_bits) - 1
            scale = 2.0 ** (-num_of_bits)

        zero_point = 0.0    
        
        q = torch.round(input_ / scale + zero_point)
        q = torch.clamp(q, qmin, qmax)

        x_q = (q - zero_point) * scale
        ctx.save_for_backward()
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None

    
class QuantizeModule(nn.Module):
    def __init__(self, num_of_bits: int = 8, signed: bool = True):
        super().__init__()
        self.num_of_bits = int(num_of_bits)
        self.signed = bool(signed)
        self.quan = quantize.apply

        # English: Register buffers so other modules (e.g., bias quant) can read them.
        if self.signed:
            base_scale = 2.0 ** (-(self.num_of_bits - 1))
        else:
            base_scale = 2.0 ** (-(self.num_of_bits))

        self.register_buffer("scale", torch.tensor(float(base_scale), dtype=torch.float32))
        self.register_buffer("zero_point", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # English: Use the same formula as quantize() for consistency; buffers are for reference.
        return self.quan(input, self.num_of_bits, self.signed)

    def extra_repr(self):
        return f'num_of_bits={self.num_of_bits}, signed={self.signed}'


class QuantConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 act_bits=16, weight_bits=16, out_bits=16, act_signed=True, weight_signed=True):
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        out_signed = act_signed | weight_signed
        self.act_quant = QuantizeModule(act_bits, act_signed)
        self.weight_quant = QuantizeModule(weight_bits, weight_signed)
        self.bias_quant = QuantizeModule(weight_bits, True)

        self.out_quant = QuantizeModule(out_bits, out_signed)
        self.out = None

    def forward(self, x):
        # x_int = self.act_quant(x)
        w_int = self.weight_quant(self.weight).to(x.device)

        bias_int = None
        if self.bias is not None:
            # 安全地獲取 scale 值
            act_scale = self.act_quant.scale 
            weight_scale = self.weight_quant.scale 
            
            bias_scale = act_scale * weight_scale
            self.bias_quant.scale.copy_(bias_scale.to(x.device))
            self.bias_quant.zero_point.copy_(torch.tensor(0.0, device=x.device))
            bias_int = self.bias_quant(self.bias)

        out = F.conv1d(x, w_int, bias_int, self.stride, self.padding,
                        self.dilation, self.groups)
        
        self.out = out.detach()
        out_q = self.out_quant(out)
        return out_q



class QuantConvTranspose1d(torch.nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 act_bits=16, weight_bits=16, out_bits=16, act_signed=True, weight_signed=True):
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
        out_signed = act_signed | weight_signed
        self.act_quant = QuantizeModule(act_bits, act_signed)
        self.weight_quant = QuantizeModule(weight_bits, weight_signed)
        self.bias_quant = QuantizeModule(weight_bits, True)

        self.out_quant = QuantizeModule(out_bits, out_signed)

        self.out = None

    def forward(self, x):
        # x_q = self.act_quant(x)
        w_q = self.weight_quant(self.weight).to(x.device)
        out = F.conv_transpose1d(x, w_q, self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups)
        
        self.out = out.detach()
        out_q = self.out_quant(out)
        return out_q


def test_basic_quantization():
    """測試基本量化功能"""
    print("=== 測試基本量化功能 ===")
    
    test_input = torch.tensor([
        0.0,
        0.123,
        -0.123,
        0.499,
        0.501,
        -0.501,
        2.5678,
        7.94,     # 靠近上限（128）
        -8.001,   # 靠近下限（-128）
    ], dtype=torch.float32)

    quant_module = QuantizeModule(num_of_bits=8, signed=True)
    x_pos_quantized = quant_module(test_input)
    
    print("原始數據: ", test_input)
    print("量化後數據: ", x_pos_quantized)


if __name__ == "__main__":

    test_basic_quantization()