from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GEEMultiSourceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.normalize = normalize

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found in {self.data_dir}")

        # 过滤整条时间序列三源（L8/S1/S2）都为 0 的样本，避免模型只学到输出票面全黑
        valid_files = []
        for f in self.files:
            try:
                with np.load(f) as data:
                    landsat = data["landsat"]
                    sentinel2 = data["sentinel2"]
                    sentinel1 = data["sentinel1"]
                    has_visible_signal = np.any(landsat != 0) or np.any(sentinel2 != 0)
                    has_any_signal = has_visible_signal or np.any(sentinel1 != 0)
                    if has_visible_signal and has_any_signal:
                        valid_files.append(f)
            except Exception:
                # 如果单个文件损坏或缺键，则直接跳过
                continue

        if not valid_files:
            raise RuntimeError(
                "All GEE multi-source samples appear to be all-zero across L8/S1/S2; "
                "please check download_gee_l8_s1_s2 outputs."
            )

        self.files = valid_files

    def __len__(self) -> int:
        return len(self.files)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        arr = arr.astype(np.float32)
        if arr.ndim == 4:
            normalized = np.zeros_like(arr, dtype=np.float32)
            for channel_idx in range(arr.shape[-1]):
                band = arr[..., channel_idx]
                v_min = np.nanmin(band)
                v_max = np.nanmax(band)
                if v_max > v_min:
                    normalized[..., channel_idx] = (band - v_min) / (v_max - v_min)
                else:
                    normalized[..., channel_idx] = band
            return normalized

        v_min = np.nanmin(arr)
        v_max = np.nanmax(arr)
        if v_max > v_min:
            return (arr - v_min) / (v_max - v_min)
        return arr

    def _frame_valid_mask(self, arr: np.ndarray) -> np.ndarray:
        return ~np.all(np.isclose(arr, 0.0), axis=(1, 2, 3))

    def _pad_time(self, x: np.ndarray, valid_mask: np.ndarray, target_T: int) -> tuple[np.ndarray, np.ndarray]:
        t, h, w, c = x.shape
        if t == target_T:
            return x, valid_mask

        if valid_mask.any():
            pad_frame = x[np.flatnonzero(valid_mask)[-1]:np.flatnonzero(valid_mask)[-1] + 1]
        else:
            pad_frame = x[-1:]

        pad = np.repeat(pad_frame, target_T - t, axis=0)
        pad_mask = np.zeros((target_T - t,), dtype=bool)
        return np.concatenate([x, pad], axis=0), np.concatenate([valid_mask, pad_mask], axis=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        with np.load(path) as data:
            landsat = data["landsat"]  # (T, H, W, C_L8)
            sentinel2 = data["sentinel2"]  # (T, H, W, C_S2)
            sentinel1 = data["sentinel1"]  # (T, H, W, C_S1)
            timestamps = data["timestamps"]  # (T,)

        T_l8, H, W, C_l8 = landsat.shape
        T_s2, H2, W2, C_s2 = sentinel2.shape
        T_s1, H3, W3, C_s1 = sentinel1.shape

        if not (H == W == H2 == W2 == H3 == W3 == self.patch_size):
            raise ValueError("All inputs must share the same patch size; adjust download script or dataset.")

        frame_valid_l8 = self._frame_valid_mask(landsat)
        frame_valid_s2 = self._frame_valid_mask(sentinel2)
        frame_valid_s1 = self._frame_valid_mask(sentinel1)

        T = int(max(T_l8, T_s2, T_s1))

        landsat, frame_valid_l8 = self._pad_time(landsat, frame_valid_l8, T)
        sentinel2, frame_valid_s2 = self._pad_time(sentinel2, frame_valid_s2, T)
        sentinel1, frame_valid_s1 = self._pad_time(sentinel1, frame_valid_s1, T)

        landsat = self._normalize(landsat)
        sentinel2 = self._normalize(sentinel2)
        sentinel1 = self._normalize(sentinel1)

        keep_mask = frame_valid_l8 | frame_valid_s2 | frame_valid_s1

        ts = np.array(timestamps, dtype=np.float64)
        if ts.shape[0] < T:
            if ts.shape[0] == 0:
                base = 0.0
            else:
                base = float(ts[-1])
            pad_ts = np.full((T - ts.shape[0],), base, dtype=np.float64)
            ts = np.concatenate([ts, pad_ts], axis=0)

        # 若存在需要过滤的时间步且不全部被过滤，则按掩码裁剪时间维
        if keep_mask.any() and not keep_mask.all():
            landsat = landsat[keep_mask]
            sentinel2 = sentinel2[keep_mask]
            sentinel1 = sentinel1[keep_mask]
            ts = ts[keep_mask]
            frame_valid_l8 = frame_valid_l8[keep_mask]
            frame_valid_s2 = frame_valid_s2[keep_mask]
            frame_valid_s1 = frame_valid_s1[keep_mask]

        landsat_tensor = torch.from_numpy(landsat).float()
        sentinel2_tensor = torch.from_numpy(sentinel2).float()
        sentinel1_tensor = torch.from_numpy(sentinel1).float()
        ts_tensor = torch.from_numpy(ts.astype(np.float32))
        frame_valid_l8_tensor = torch.from_numpy(frame_valid_l8.astype(np.bool_))
        frame_valid_s2_tensor = torch.from_numpy(frame_valid_s2.astype(np.bool_))
        frame_valid_s1_tensor = torch.from_numpy(frame_valid_s1.astype(np.bool_))

        if ts.size > 0:
            valid_start = float(ts[0])
            valid_end = float(ts[-1])
        else:
            valid_start = 0.0
            valid_end = 0.0

        return {
            "source_data": {
                "landsat": landsat_tensor,
                "sentinel1": sentinel1_tensor,
                "sentinel2": sentinel2_tensor,
            },
            "timestamps": {
                "landsat": ts_tensor,
                "sentinel1": ts_tensor,
                "sentinel2": ts_tensor,
            },
            "frame_valid_mask": {
                "landsat": frame_valid_l8_tensor,
                "sentinel1": frame_valid_s1_tensor,
                "sentinel2": frame_valid_s2_tensor,
            },
            "valid_period": (valid_start, valid_end),
        }


def create_gee_multisource_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: int = 256,
    normalize: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    dataset = GEEMultiSourceDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        normalize=normalize,
    )

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_names = list(batch[0]["source_data"].keys())
        B = len(batch)

        collated_sources: Dict[str, torch.Tensor] = {}
        collated_timestamps: Dict[str, torch.Tensor] = {}
        collated_frame_masks: Dict[str, torch.Tensor] = {}

        for src in source_names:
            tensors = [sample["source_data"][src] for sample in batch]
            masks = [sample["frame_valid_mask"][src] for sample in batch]
            T_max = max(t.shape[0] for t in tensors)
            padded: List[torch.Tensor] = []
            padded_masks: List[torch.Tensor] = []
            for t, mask in zip(tensors, masks):
                if t.shape[0] < T_max:
                    T, H, W, C = t.shape
                    pad_frame = t[-1:].repeat(T_max - T, 1, 1, 1)
                    pad = pad_frame if T > 0 else torch.zeros(T_max - T, H, W, C, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=0)
                    mask = torch.cat([mask, torch.zeros(T_max - mask.shape[0], dtype=torch.bool)], dim=0)
                padded.append(t)
                padded_masks.append(mask)
            collated_sources[src] = torch.stack(padded)
            collated_frame_masks[src] = torch.stack(padded_masks)

            ts_list = [sample["timestamps"][src] for sample in batch]
            T_ts_max = max(len(ts) for ts in ts_list)
            ts_padded: List[torch.Tensor] = []
            for ts in ts_list:
                if ts.shape[0] < T_ts_max:
                    last = ts[-1] if ts.shape[0] > 0 else torch.tensor(0.0, dtype=ts.dtype)
                    pad_ts = torch.full((T_ts_max - ts.shape[0],), float(last), dtype=ts.dtype)
                    ts = torch.cat([ts, pad_ts])
                ts_padded.append(ts)
            collated_timestamps[src] = torch.stack(ts_padded)

        valid_periods = [sample["valid_period"] for sample in batch]

        return {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "frame_valid_mask": collated_frame_masks,
            "valid_periods": valid_periods,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
