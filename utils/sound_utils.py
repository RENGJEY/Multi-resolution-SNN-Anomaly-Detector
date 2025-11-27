import os
import torch
import librosa
import pandas as pd
import numpy as np
import scipy.signal as sp
from tqdm.notebook import tqdm
from snntorch import spikegen

from utils.misc import dump_pickle, load_pickle


def load_sound_file(path, mono=False, channel=0):
    signal, sr = librosa.load(path, sr=None, mono=mono)
    if signal.ndim < 2:
        sound_file = signal, sr
    else:
        sound_file = signal[channel, :], sr

    return sound_file

def poisson_encode(x_prob):
    # x_prob in [0,1], return Bernoulli spikes (0/1) with P= x_prob
    return torch.bernoulli(x_prob)

def normalize_features(feats, mode):
    if feats.size == 0:
        return feats
    if mode == "per_frame":  # 每筆樣本內 z-score
        mu = feats.mean(axis=1, keepdims=True)
        sd = feats.std(axis=1, keepdims=True)
        return (feats - mu) / (sd + 1e-8)
    elif mode == "per_dim":  # 每個頻帶
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True)
        return (feats - mu) / (sd + 1e-8)
    elif mode == "minmax":
        minv = feats.min(axis=1, keepdims=True)
        maxv = feats.max(axis=1, keepdims=True)
        return (feats - minv) / (maxv - minv + 1e-8)
    else:
        return feats  # 不做 normalizatio
    

def _stack_frames_strided(logmel: np.ndarray, frames: int) -> np.ndarray:
    """logmel [n_mels, T] -> [T-frames+1, frames*n_mels] float32"""
    n_mels, T = logmel.shape
    if T < frames:
        return np.empty((0, n_mels * frames), np.float32)
    s0, s1 = logmel.strides  # strides for [n_mels, time]
    view = np.lib.stride_tricks.as_strided(
        logmel, shape=(n_mels, frames, T - frames + 1),
        strides=(s0, s1, s1), writeable=False
    )
    view = np.moveaxis(view, (0,1,2), (2,1,0))  # [T-frames+1, frames, n_mels]
    return view.reshape(view.shape[0], -1).astype(np.float32)


def extract_sound_features(
    signal, sr, n_fft=1024, hop_length=512, n_mels=64, frames=5, fmax=3000, per_frame_zscore=False
):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax
    )
    logmel  = librosa.power_to_db(mel_spectrogram, ref=np.max)
    feats = _stack_frames_strided(logmel, frames)

    # 特徵保留絕對能量大小資訊: per_frame_zscore=False
    feats = normalize_features(feats, mode="minmax" if per_frame_zscore else "none")

    return feats


def extract_gsensor_features(
    signal, sr, n_fft=2048, hop_length=512, n_mels=64, frames=5, fmax=3000, per_frame_zscore=False
):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=fmax, power=2.0
    )
    logmel  = librosa.power_to_db(mel_spectrogram, ref=np.max)
    feats = _stack_frames_strided(logmel, frames)

    # 特徵保留絕對能量大小資訊: per_frame_zscore=False
    feats = normalize_features(feats, mode="minmax" if per_frame_zscore else "none")

    return feats

def generate_sound_dataset(
        files_list, time_step, n_fft=2048, hop_length=512, n_mels=64, frames=5, fmax=3000, 
        normalization=False, progress=False
    ):

    all_features = []    
    dims = n_mels * frames
    iterator = range(len(files_list))
    if progress:
        iterator = tqdm(iterator, desc="Gen feats", leave=False, dynamic_ncols=True)

    for index in iterator:
        signal, sr = load_sound_file(files_list[index])
        features = extract_sound_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            frames=frames,
            fmax=fmax,
            per_frame_zscore=normalization
        )
        all_features.append(features)

    dataset = np.vstack(all_features)  # (total_samples, dims)

    probs = torch.from_numpy(dataset).float().clamp_(0.0, 1.0)  # (B, F)

    # Rate coding → (T, B, F) → (B, T, F)
    spikes_dataset = spikegen.rate(probs, num_steps=time_step).transpose(0, 1).contiguous()


    probs_time = probs.unsqueeze(1).expand(-1, time_step, -1).contiguous()

    return spikes_dataset, probs_time

def generate_gsensor_dataset(
        files_list, time_step, n_fft=1024, hop_length=512, n_mels=64, frames=5, fmax=3000, 
        normalization=False, progress=False
    ):

    dims = n_mels * frames
    all_features = []

    iterator = range(len(files_list))
    if progress:
        iterator = tqdm(iterator, desc="Gen feats", leave=False, dynamic_ncols=True)

    for index in iterator:
        signal, sr = load_sound_file(files_list[index])
        
        features = extract_gsensor_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            frames=frames,
            fmax=fmax,
            per_frame_zscore=normalization
        )
        all_features.append(features)

    dataset = np.vstack(all_features)  # (total_samples, dims)

    probs = torch.from_numpy(dataset).float().clamp_(0.0, 1.0)  # (B, F)

    # Rate coding → (T, B, F)
    spikes_dataset = spikegen.rate(probs, num_steps=time_step).transpose(0, 1).contiguous()

    probs_time = probs.unsqueeze(1).expand(-1, time_step, -1).contiguous()

    return spikes_dataset, probs_time

#######################################################################################
############################## datasets build utils ###################################
#######################################################################################

# ====== 1) 基本設定打包（可改成 dataclass 亦可） ======
def make_build_cfg(n_fft, time_step, hop_length_sound, hop_length_gsensor,
                   n_mels1, frames1, n_mels2, frames2,
                   fmax_sound=3000, fmax_g=1800, normalization=True):
    """Collects all config values into a dict for easy passing."""
    return dict(
        n_fft=n_fft,
        time_step=time_step,
        hop_length_sound=hop_length_sound,
        hop_length_gsensor=hop_length_gsensor,
        n_mels1=n_mels1,
        frames1=frames1,
        n_mels2=n_mels2,
        frames2=frames2,
        fmax_sound=fmax_sound,
        fmax_g=fmax_g,
        normalization=normalization
    )


# ====== 2) 單檔處理（不保存，只回傳結果） ======
def process_one_pair(snd_path, g_path, cfg):
    """Compute (spikes, probs) for one sound/gsensor pair and align N."""
    # --- compute per-file features ---
    snd_spk, snd_prb = generate_sound_dataset(
        [snd_path],
        n_fft=cfg["n_fft"],
        time_step=cfg["time_step"],
        hop_length=cfg["hop_length_sound"],
        n_mels=cfg["n_mels1"],
        frames=cfg["frames1"],
        fmax=cfg["fmax_sound"],
        progress=False,
        normalization=cfg["normalization"],
    )
    gs_spk, gs_prb = generate_gsensor_dataset(
        [g_path],
        time_step=cfg["time_step"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length_gsensor"],
        n_mels=cfg["n_mels2"],
        frames=cfg["frames2"],
        fmax=cfg["fmax_g"],
        progress=False,
        normalization=cfg["normalization"],
    )

    # --- skip if empty ---
    if snd_spk.size == 0 or gs_spk.size == 0:
        return None  # indicate unusable

    # --- align N_i across modalities ---
    N_i = min(snd_spk.shape[0], gs_spk.shape[0])
    if N_i <= 0:
        return None

    # --- sanity checks on time dimension ---
    T = cfg["time_step"]
    assert snd_spk.shape[1] == T and gs_spk.shape[1] == T, "spike time_step mismatch"
    assert snd_prb.shape[1] == T and gs_prb.shape[1] == T, "prob time_step mismatch"

    # --- crop to aligned N_i ---
    snd_spk, gs_spk = snd_spk[:N_i], gs_spk[:N_i]
    snd_prb, gs_prb = snd_prb[:N_i], gs_prb[:N_i]

    # --- concatenate by feature dimension ---
    comb_spk = np.concatenate([snd_spk, gs_spk], axis=-1)  # (N_i, T, F_s+F_g)
    comb_prb = np.concatenate([snd_prb, gs_prb], axis=-1)  # (N_i, T, F_s+F_g)

    return comb_spk, comb_prb


# ====== 3) PASS-1：掃描檔案，計算 N_total / F_total 與 metadata ======
def pass1_scan(split_name, file_list, out_dir, cfg, gsensor_root, sound_root):
    """First pass: compute sizes & metadata; DO NOT keep big arrays."""
    # output metadata paths
    counts_path    = os.path.join(out_dir, f"{split_name}_frame_counts.pkl")
    slices_path    = os.path.join(out_dir, f"{split_name}_file_slices.pkl")
    filenames_path = os.path.join(out_dir, f"{split_name}_filenames.pkl")

    # build counterpart gsensor paths
    gsensor_files = [
        os.path.join(gsensor_root, os.path.relpath(p, sound_root))
        for p in file_list
    ]

    frame_counts, file_slices = [], []
    filenames = list(file_list)

    N_total = 0
    F_total = None
    T_seen  = cfg["time_step"]

    print(f"[PASS-1] Scanning {split_name.upper()} ...")
    for snd_path, g_path in tqdm(list(zip(file_list, gsensor_files)), total=len(file_list), dynamic_ncols=True, leave=False):
        res = process_one_pair(snd_path, g_path, cfg)
        if res is None:
            # unusable, record empty slice
            frame_counts.append(0)
            file_slices.append((N_total, N_total))
            continue

        comb_spk, comb_prb = res
        N_i, T_i, F_i = comb_spk.shape

        # infer F_total from first valid file
        if F_total is None:
            F_total = F_i
        else:
            assert F_i == F_total, "Feature dimension inconsistent across files."

        # record slices and counts
        start, end = N_total, N_total + N_i
        file_slices.append((start, end))
        frame_counts.append(N_i)
        N_total = end

    if N_total == 0 or F_total is None:
        raise RuntimeError(f"No usable files for split {split_name} in PASS-1.")

    # save metadata now (optional)
    dump_pickle(counts_path,    frame_counts)
    dump_pickle(slices_path,    file_slices)
    dump_pickle(filenames_path, filenames)

    meta = dict(
        frame_counts=frame_counts,
        file_slices=file_slices,
        filenames=filenames,
        N_total=N_total,
        T=T_seen,
        F_total=F_total,
        gsensor_files=gsensor_files
    )
    return meta


# ====== 4) 配置陣列（一次配置） ======
def allocate_arrays(N_total, T, F_total, dtype=np.float32):
    """Allocate once to avoid giant concatenation copies."""
    spk = np.empty((N_total, T, F_total), dtype=dtype)
    prb = np.empty((N_total, T, F_total), dtype=dtype)
    return spk, prb


# ====== 5) PASS-2：填入（逐檔填 slice） ======
def pass2_fill(split_name, file_list, meta, cfg, gsensor_root, sound_root, out_dir):
    """Second pass: fill pre-allocated arrays by slices."""
    out_path     = os.path.join(out_dir, f"{split_name}_data_combined.pkl")
    out_prb_path = os.path.join(out_dir, f"{split_name}_probs_combined.pkl")

    N_total, T, F_total = meta["N_total"], meta["T"], meta["F_total"]
    file_slices = meta["file_slices"]

    combined_spk, combined_prb = allocate_arrays(N_total, T, F_total)

    print(f"[PASS-2] Filling arrays for {split_name.upper()} ...")
    gsensor_files = [
        os.path.join(gsensor_root, os.path.relpath(p, sound_root))
        for p in file_list
    ]

    for (snd_path, g_path), (start, end) in tqdm(
        zip(zip(file_list, gsensor_files), file_slices),
        total=len(file_list), dynamic_ncols=True, leave=False
    ):
        if end <= start:
            continue  # empty slice

        # recompute pair (stateless, avoids holding many arrays)
        res = process_one_pair(snd_path, g_path, cfg)
        if res is None:
            raise RuntimeError(f"PASS-2 found unusable pair that was usable in PASS-1: {snd_path}, {g_path}")

        comb_spk, comb_prb = res
        N_i = end - start
        assert comb_spk.shape[0] >= N_i, "Inconsistent N_i between passes."

        # fill slices
        combined_spk[start:end, :, :] = comb_spk[:N_i]
        combined_prb[start:end, :, :] = comb_prb[:N_i]

    # save combined arrays
    dump_pickle(out_path,     combined_spk)
    dump_pickle(out_prb_path, combined_prb)

    return combined_spk, combined_prb


# ====== 6) 主函式：兩趟法建置並保存 metadata ======
def build_split_dataset(split_name, file_list, out_dir, cfg, gsensor_root, sound_root):
    """
    Orchestrator for two-pass build:
      - pass1: scan to get sizes & metadata
      - pass2: allocate once and fill slices
      - save arrays & metadata
    """
    # output paths for quick reuse
    out_path       = os.path.join(out_dir, f"{split_name}_data_combined.pkl")
    out_prb_path   = os.path.join(out_dir, f"{split_name}_probs_combined.pkl")
    counts_path    = os.path.join(out_dir, f"{split_name}_frame_counts.pkl")
    slices_path    = os.path.join(out_dir, f"{split_name}_file_slices.pkl")
    filenames_path = os.path.join(out_dir, f"{split_name}_filenames.pkl")

    # fast path if everything exists
    if all(os.path.exists(p) for p in [out_path, out_prb_path, counts_path, slices_path, filenames_path]):
        print(f"{split_name.capitalize()} data & metadata already exist, loading from file...")
        return load_pickle(out_path), load_pickle(out_prb_path)

    # pass-1
    meta = pass1_scan(split_name, file_list, out_dir, cfg, gsensor_root, sound_root)

    # pass-2
    combined_spk, combined_prb = pass2_fill(split_name, file_list, meta, cfg, gsensor_root, sound_root, out_dir)

    # save metadata (already saved in pass1; safe to overwrite if desired)
    dump_pickle(counts_path,    meta["frame_counts"])
    dump_pickle(slices_path,    meta["file_slices"])
    dump_pickle(filenames_path, meta["filenames"])

    print(f"Saved {split_name} spikes to: {out_path}")
    print(f"Saved {split_name} probs  to: {out_prb_path}")
    print(f"Saved {split_name} frame_counts to: {counts_path}")
    print(f"Saved {split_name} file_slices to: {slices_path}")
    print(f"Saved {split_name} filenames to: {filenames_path}")
    print(f"{split_name.capitalize()} combined shapes: spikes={combined_spk.shape}, probs={combined_prb.shape}")

    return combined_spk, combined_prb