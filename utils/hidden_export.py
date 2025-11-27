from typing import Dict, Optional
import os
import torch
import utils.model_quan2 as mq

@torch.no_grad()
def export_hidden_means_qpot(
    model: torch.nn.Module,
    path: str,
    *,
    quantizer: Optional[mq.QuantizePOTModule] = None,
    num_bits: int = 8,
    signed: bool = True,
    cast_to_int8: bool = True,
    also_save_npz: bool = False
):
    """
    Export model.get_hidden_mean() as POT-quantized package to disk.
    - model: must implement .get_hidden_mean() -> Dict[str, Tensor or None]
    - path : .pth file to store a dict package (torch.save)
    - quantizer: optional external QuantizePOTModule instance to reuse.
    """
    # 1) Prepare a quantizer (reuse or create)
    q = quantizer if quantizer is not None else mq.QuantizePOTModule(num_bits=num_bits, signed=signed)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2) Pull means from model
    hidden_means: Dict[str, Optional[torch.Tensor]] = model.get_hidden_mean()

    pkg = {
        "_format": "hidden_mean_qpot_v1",
        "num_bits": int(q.num_bits),
        "signed": bool(q.signed),
        "scale": float(q.scale.item()),
        "zero_point": int(q.zero_point.item()),  # 0 for POT
        "items": {}
    }

    # 3) Per-layer quantize → save integers + (optional) dequant
    for name, x in hidden_means.items():
        if x is None:
            continue

        x = x.detach().to("cpu").float()  # (1, C, L) or any shape

        # Forward (in no_grad): returns x_q and fills buffers
        _ = q(x)

        q_int = q.q_int_buf.clone()        # int32 by default
        deq   = q.dequant_buf.clone()      # float32

        if cast_to_int8:
            q_int = q_int.to(torch.int8 if q.signed else torch.uint8)

        pkg["items"][name] = {
            "shape": tuple(x.shape),
            "q": q_int,
            "dequant": deq,  # keep for convenience; can be removed to save space
            "scale": float(q.scale.item()),
            "zero_point": int(q.zero_point.item()),
            "signed": bool(q.signed),
        }

    torch.save(pkg, path)

    # 4) Optional .npz mirror for non-PyTorch consumers
    if also_save_npz:
        import numpy as np
        base, _ = os.path.splitext(path)
        npz_path = base + ".npz"
        npz_items = {}
        for name, it in pkg["items"].items():
            npz_items[f"{name}/q"]       = it["q"].cpu().numpy()
            npz_items[f"{name}/dequant"] = it["dequant"].cpu().numpy()
            npz_items[f"{name}/shape"]   = np.array(it["shape"], dtype=np.int64)
            npz_items[f"{name}/scale"]   = np.array(it["scale"], dtype=np.float32)
            npz_items[f"{name}/zp"]      = np.array(it["zero_point"], dtype=np.int32)
            npz_items[f"{name}/signed"]  = np.array(int(it["signed"]), dtype=np.int8)
        np.savez_compressed(
            npz_path,
            _format=pkg["_format"],
            num_bits=np.int32(pkg["num_bits"]),
            signed=np.int8(int(pkg["signed"])),
            scale=np.float32(pkg["scale"]),
            zero_point=np.int32(pkg["zero_point"]),
            **npz_items
        )


@torch.no_grad()
def load_hidden_from_qpot_package(
    model: torch.nn.Module,
    path: str,
    *,
    batch_size: int,
    device: torch.device,
    use_dequant: bool = True
):
    """
    Load a qpot export package (.pth) and broadcast into the model via model.load_hidden(...).
    - If use_dequant=True, use stored float 'dequant'.
    - Else reconstruct by (q * scale), where zp=0 in POT.
    """
    pkg = torch.load(path, map_location="cpu")
    items = pkg["items"]
    loaded = {}

    for name, it in items.items():
        if use_dequant and ("dequant" in it):
            x = it["dequant"].to(torch.float32)
        else:
            q_tensor = it["q"].to(torch.float32)
            x = q_tensor * float(it.get("scale", pkg["scale"]))  # zp=0 for POT
        loaded[name] = x.view(it["shape"]).contiguous()

    # model must implement .load_hidden(hidden: dict, batch_size: int, device: torch.device)
    model.load_hidden(loaded, batch_size=batch_size, device=device)
