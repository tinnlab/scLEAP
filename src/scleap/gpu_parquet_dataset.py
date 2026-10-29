#!/usr/bin/env python3
# gpu_parquet_window_loader.py
# Windowed, zero-copy Parquet -> PyTorch (DDP-safe, low-memory, no dask),
# with carryover so batches are consistently full, plus --drop-last.

from __future__ import annotations
import os
import argparse
import random
import warnings
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import IterableDataset

import cupy as cp
import cudf
import rmm
from tqdm import tqdm


# -----------------------------
# DLPack helpers
# -----------------------------

def _to_torch(a: cp.ndarray) -> torch.Tensor:
    """Zero-copy CuPy -> Torch CUDA tensor via DLPack."""
    try:
        return torch.utils.dlpack.from_dlpack(a)
    except TypeError:
        return torch.utils.dlpack.from_dlpack(a.toDlpack())


def _values(s: cudf.Series) -> cp.ndarray:
    """Return CuPy device array for a numeric cudf Series."""
    return s.values


# -----------------------------
# LIST column helpers
# -----------------------------

def _get_list_children(s: cudf.Series) -> Tuple[cudf.Series, cudf.Series]:
    """
    Return (offsets, values) child columns for a LIST cudf Series.

    offsets has length n_rows + 1
    values contains the flattened leaf values
    """
    col = s._column
    children = getattr(col, "children", None) or getattr(col, "base_children", None)
    if not children or len(children) < 2:
        raise TypeError(f"Column {s.name!r} is not a LIST column")
    offs = cudf.Series._from_column(children[0])
    vals = cudf.Series._from_column(children[1])
    return offs, vals


def _slice_list_batch(
    s_full: cudf.Series,
    start: int,
    stop: int,
) -> Tuple[cudf.Series, cudf.Series]:
    """
    Slice rows [start, stop) from a LIST column and return batch-local
    offsets and values. Returned offsets are rebased to start at 0.

    This is critical because newer cuDF versions may keep LIST child values
    pointing to a larger backing buffer after slicing.
    """
    offs, vals = _get_list_children(s_full)

    start_off = int(offs.iloc[start])
    stop_off = int(offs.iloc[stop])

    offs_view = offs.iloc[start:stop + 1] - start_off
    vals_view = vals.iloc[start_off:stop_off]
    return offs_view, vals_view


def _fixed_matrix_or_none(
    offs_cu: cp.ndarray,
    vals_cu: cp.ndarray,
    rows: int,
) -> Optional[cp.ndarray]:
    """
    If the LIST column is fixed-width for this batch, return a 2D matrix
    of shape (rows, L). Otherwise return None.
    """
    if rows == 0:
        return vals_cu.reshape(0, 0)

    diffs = cp.diff(offs_cu)
    if int(diffs.min()) == int(diffs.max()):
        L = int(diffs[0])
        if L < 0:
            return None
        return vals_cu.reshape(rows, L)
    return None


def _list_batch_to_torch(
    s_full: cudf.Series,
    start: int,
    stop: int,
    col_name: str,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Materialize rows [start, stop) from a LIST column into either:
      - a dense torch.Tensor if fixed-width
      - a dict {"values": ..., "offsets": ...} if ragged

    Uses explicit child slicing so behavior is stable across cuDF versions.
    """
    rows = stop - start
    offs, vals = _slice_list_batch(s_full, start, stop)

    if not cudf.api.types.is_numeric_dtype(vals.dtype):
        raise TypeError(f"List column {col_name} leaves must be numeric, got {vals.dtype}")

    offs_cu = cp.asarray(_values(offs)).reshape(-1)
    vals_cu = cp.asarray(_values(vals)).reshape(-1)

    if rows > 0:
        diffs = cp.diff(offs_cu)
        if int(diffs.min()) == int(diffs.max()):
            L = int(diffs[0])
            expected = rows * L
            if vals_cu.size != expected:
                raise ValueError(
                    f"LIST column {col_name!r} mismatch after slicing: "
                    f"rows={rows}, L={L}, expected vals={expected}, got vals={vals_cu.size}"
                )

    mat = _fixed_matrix_or_none(offs_cu, vals_cu, rows)
    if mat is not None:
        return _to_torch(mat)

    return {
        "values": _to_torch(vals_cu),
        "offsets": _to_torch(offs_cu),
    }


# -----------------------------
# Rank/world helpers
# -----------------------------

def _rank_world() -> Tuple[int, int]:
    r, w = 0, 1
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass

    r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    w = int(os.environ.get("WORLD_SIZE", "1"))
    return r, w


# -----------------------------
# Optional RMM pool (default: OFF)
# -----------------------------

def _init_rmm(pool: str = "off") -> bool:
    """
    Return True if we configured an RMM pool and hooked CuPy to it, else False.

    When pool == 'off', use no RMM pool and leave CuPy on its own allocator.
    """
    pool = (pool or "off").lower()
    using_rmm_pool = False

    if pool in ("off", "none", "0"):
        try:
            rmm.reinitialize(pool_allocator=False)
        except Exception as e:
            warnings.warn(f"RMM no-pool init failed ({e}); continuing.")
        return using_rmm_pool

    def _bytes(s: str) -> int:
        s = s.lower().strip()
        mult = 1
        if s.endswith(("gb", "g")):
            mult = 1 << 30
            s = s.rstrip("gbg")
        elif s.endswith(("mb", "m")):
            mult = 1 << 20
            s = s.rstrip("mbm")
        elif s.endswith(("kb", "k")):
            mult = 1 << 10
            s = s.rstrip("kbk")
        return int(float(s) * mult)

    want = _bytes(pool)
    free, _ = cp.cuda.runtime.memGetInfo()
    want = max(256 << 20, min(int(0.8 * free), (want // 256) * 256))

    try:
        rmm.reinitialize(pool_allocator=True, initial_pool_size=want)
        using_rmm_pool = True
    except Exception as e:
        warnings.warn(f"RMM fixed pool {want}B failed: {e}. Falling back to no-pool.")
        try:
            rmm.reinitialize(pool_allocator=False)
        except Exception:
            pass
        using_rmm_pool = False

    if using_rmm_pool:
        try:
            from rmm.allocators.cupy import rmm_cupy_allocator
            cp.cuda.set_allocator(rmm_cupy_allocator)
        except Exception:
            try:
                cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
            except Exception as e:
                warnings.warn(f"Failed to hook CuPy to RMM pool: {e}")
                using_rmm_pool = False

    return using_rmm_pool


# -----------------------------
# Dataset
# -----------------------------

class WindowParquetDataset(IterableDataset):
    """
    DDP-safe, low-memory parquet reader:
      - shards files across ranks
      - reads each file in row windows with cudf.read_parquet(skip_rows=.., nrows=..)
      - shuffles window order and rows within each window
      - uses carryover so batches stay full across window/file boundaries
      - yields zero-copy CUDA tensors for scalar and LIST columns

    __len__ returns the number of batches this rank will yield:
      floor(total_rows/batch) if drop_last else ceil(total_rows/batch)
    """

    def __init__(
        self,
        files: Sequence[str],
        columns: Sequence[str],
        list_columns: Optional[Sequence[str]] = None,
        label_columns: Optional[Sequence[str]] = None,
        casts: Optional[Dict[str, str]] = None,
        batch_size: int = 1024,
        window_rows: int = 65536,
        shuffle: bool = True,
        part_shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        device_id: Optional[int] = None,
        rmm_pool: str = "off",
    ):
        super().__init__()
        self.files = list(files)
        self.columns = list(columns)
        self.list_cols = set(list_columns or [])
        self.label_cols = set(label_columns or [])
        self.casts = dict(casts or {})

        for c in self.list_cols | self.label_cols:
            if c not in self.columns:
                self.columns.append(c)

        self.batch = int(batch_size)
        self.window = int(window_rows)
        self.shuffle = bool(shuffle)
        self.part_shuffle = bool(part_shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.device_id = device_id
        self.rmm_pool = rmm_pool

        self.rank, self.world = _rank_world()

        self._len_cache: Optional[int] = None
        self._len_cache_key: Optional[Tuple[int, int, int, bool]] = None
        self._using_rmm_pool: bool = False

    # ---------- helpers for length ----------

    def _sharded_files(self) -> List[str]:
        return self.files[self.rank::self.world]

    def _file_num_rows(self, path: str) -> int:
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        except Exception:
            tmp = cudf.read_parquet(path, columns=[self.columns[0]])
            n = len(tmp)
            del tmp
            return n

    def _compute_num_rows_for_rank(self) -> int:
        total = 0
        for p in self._sharded_files():
            total += self._file_num_rows(p)
        return int(total)

    def __len__(self) -> int:
        self.rank, self.world = _rank_world()
        key = (self.rank, self.world, self.batch, self.drop_last)

        if self._len_cache is not None and self._len_cache_key == key:
            return self._len_cache

        rows = self._compute_num_rows_for_rank()
        batches = (rows // self.batch) if self.drop_last else ((rows + self.batch - 1) // self.batch)
        self._len_cache = int(batches)
        self._len_cache_key = key
        return self._len_cache

    # ---------- device / memory setup ----------

    def _init_device(self):
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            raise RuntimeError("No visible CUDA devices.")

        lr = os.environ.get("LOCAL_RANK")
        if self.device_id is None:
            self.device_id = int(lr) % n if lr is not None else 0
        if not (0 <= self.device_id < n):
            self.device_id = 0

        cp.cuda.Device(self.device_id).use()
        torch.cuda.set_device(self.device_id)

    # ---------- lifecycle ----------

    def close(self):
        """Free CuPy blocks and drop RMM pool if one was used."""
        try:
            cp.cuda.Device(self.device_id or 0).synchronize()
        except Exception:
            pass

        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        if self._using_rmm_pool:
            try:
                rmm.reinitialize(pool_allocator=False)
            except Exception:
                pass
            try:
                mp = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(mp.malloc)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---------- row conversion helpers ----------

    def _pack_batch(
        self,
        gdf_source: cudf.DataFrame,
        start: int,
        stop: int,
    ) -> Union[
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        Tuple[
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
            Union[torch.Tensor, Dict[str, torch.Tensor]],
        ],
    ]:
        """
        Convert gdf_source.iloc[start:stop] into model-ready tensors.
        LIST columns are always sliced explicitly from gdf_source.
        """
        gb = gdf_source.iloc[start:stop]
        out: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}

        # Scalar features
        for c in self.columns:
            if c in self.list_cols or c not in gb._data:
                continue
            if c in self.label_cols:
                continue
            if not cudf.api.types.is_numeric_dtype(gb[c].dtype):
                raise TypeError(f"Scalar column {c} must be numeric, got {gb[c].dtype}")
            out[c] = _to_torch(_values(gb[c]))

        # LIST features
        for c in self.list_cols:
            if c not in gdf_source._data:
                continue
            out[c] = _list_batch_to_torch(gdf_source[c], start, stop, c)

        # Labels
        if self.label_cols:
            labels = {
                k: (_to_torch(_values(gb[k])) if k not in self.list_cols else out[k])
                for k in self.label_cols
            }
            feats = {k: v for k, v in out.items() if k not in self.label_cols}
            if len(labels) == 1:
                labels = next(iter(labels.values()))
            return feats, labels

        return out

    # ---------- iteration ----------

    def __iter__(self) -> Iterator[
        Union[
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
            Tuple[
                Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
                Union[torch.Tensor, Dict[str, torch.Tensor]],
            ],
        ]
    ]:
        self.rank, self.world = _rank_world()

        self._init_device()
        self._using_rmm_pool = _init_rmm(self.rmm_pool)

        files = self.files[self.rank::self.world]
        if self.part_shuffle:
            rnd = random.Random(self.seed + self.rank * 17)
            rnd.shuffle(files)

        carry: Optional[cudf.DataFrame] = None

        try:
            for path in files:
                try:
                    import pyarrow.parquet as pq
                    num_rows = pq.ParquetFile(path).metadata.num_rows
                except Exception:
                    tmp = cudf.read_parquet(path, columns=[self.columns[0]])
                    num_rows = len(tmp)
                    del tmp

                window_starts = list(range(0, num_rows, self.window))
                if self.part_shuffle:
                    rnd = random.Random(self.seed ^ (hash(path) & 0xFFFFFFFF))
                    rnd.shuffle(window_starts)

                for start_row in window_starts:
                    nrows = min(self.window, num_rows - start_row)

                    gdf = cudf.read_parquet(
                        path,
                        columns=self.columns,
                        skip_rows=start_row,
                        nrows=nrows,
                    )

                    if carry is not None and len(carry) > 0:
                        gdf = cudf.concat([carry, gdf], ignore_index=True)
                        carry = None

                    for c, dt in self.casts.items():
                        if c in gdf._data:
                            gdf[c] = gdf[c].astype(dt)

                    if self.shuffle and len(gdf) > 1:
                        try:
                            idx = cp.random.permutation(len(gdf))
                        except Exception:
                            idx = cp.arange(len(gdf))
                        gdf = gdf.take(cudf.Series(idx))
                        gdf = gdf.reset_index(drop=True)

                    n = len(gdf)
                    bs = self.batch
                    last_full_end = (n // bs) * bs

                    for i in range(0, last_full_end, bs):
                        yield self._pack_batch(gdf, i, i + bs)

                    rem = gdf.iloc[last_full_end:]
                    carry = rem if len(rem) > 0 else None

                    del gdf

            if carry is not None and len(carry) > 0 and not self.drop_last:
                yield self._pack_batch(carry, 0, len(carry))

        finally:
            self.close()


# -----------------------------
# Thin loader wrapper
# -----------------------------

def build_loader(
    files: Sequence[str],
    columns: Sequence[str],
    list_columns: Optional[Sequence[str]] = None,
    label_columns: Optional[Sequence[str]] = None,
    casts: Optional[Dict[str, str]] = None,
    batch_size: int = 1024,
    window_rows: int = 65536,
    shuffle: bool = True,
    part_shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 0,
    device_id: Optional[int] = None,
    rmm_pool: str = "off",
) -> torch.utils.data.DataLoader:
    ds = WindowParquetDataset(
        files=files,
        columns=columns,
        list_columns=list_columns,
        label_columns=label_columns,
        casts=casts,
        batch_size=batch_size,
        window_rows=window_rows,
        shuffle=shuffle,
        part_shuffle=part_shuffle,
        drop_last=drop_last,
        seed=seed,
        device_id=device_id,
        rmm_pool=rmm_pool,
    )
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0, pin_memory=False)


# -----------------------------
# Small benchmark
# -----------------------------

def _unpack(b):
    if isinstance(b, tuple) and len(b) == 2:
        return b[0], b[1]
    if isinstance(b, list) and len(b) == 2:
        return b[0], b[1]
    if isinstance(b, dict):
        return b, None
    if isinstance(b, torch.Tensor):
        return {"_": b}, None
    raise TypeError(f"Unexpected batch type: {type(b)}")


def _rows(b) -> int:
    feats, labels = _unpack(b)
    if isinstance(labels, torch.Tensor):
        return labels.shape[0]
    if isinstance(feats, dict):
        any_feat = next(iter(feats.values()))
        if isinstance(any_feat, dict):
            offs = any_feat["offsets"]
            return (offs.shape[0] - 1) if hasattr(offs, "shape") else (len(offs) - 1)
        if isinstance(any_feat, torch.Tensor):
            return any_feat.shape[0]
    if isinstance(feats, torch.Tensor):
        return feats.shape[0]
    raise TypeError(f"Cannot infer rows from batch type: {type(b)}")


def run_bench(loader, measure_batches: int):
    import time

    it = iter(loader)
    for _ in range(min(5, measure_batches)):
        _ = next(it)
        torch.cuda.synchronize()

    n_rows = 0
    t0 = time.perf_counter()
    for _ in range(measure_batches):
        b = next(it)
        torch.cuda.synchronize()
        n_rows += _rows(b)
    dt = time.perf_counter() - t0
    print(f"[BENCH] {n_rows:,} rows in {dt:.3f}s  =>  {n_rows/dt:,.0f} rows/s")


# -----------------------------
# CLI
# -----------------------------

def _csv(s: Optional[str]) -> List[str]:
    return [] if not s else [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser("Windowed Parquet -> Torch (DDP-safe, carryover batching)")
    ap.add_argument("--parquet-root", required=True, type=str)
    ap.add_argument("--columns", required=True, type=str, help="features+labels, comma-separated")
    ap.add_argument("--list-columns", default="", type=str, help="LIST feature columns")
    ap.add_argument("--label-columns", default="", type=str, help="label columns")
    ap.add_argument("--cast", default="", type=str, help="e.g. ontology_int=int64")
    ap.add_argument("--batch-size", default=2048, type=int)
    ap.add_argument("--window-rows", default=65536, type=int)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--part-shuffle", action="store_true", default=True)
    ap.add_argument("--no-part-shuffle", dest="part_shuffle", action="store_false")
    ap.add_argument("--drop-last", action="store_true", default=False, help="drop final short batch")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device-id", default=None, type=int)
    ap.add_argument("--rmm-pool", default="off", type=str, help="'off' or like '8GB'")
    ap.add_argument("--bench", default=0, type=int)
    args = ap.parse_args()

    files = [
        os.path.join(args.parquet_root, f)
        for f in os.listdir(args.parquet_root)
        if f.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"No parquet files in {args.parquet_root}")

    columns = _csv(args.columns)
    list_cols = _csv(args.list_columns)
    label_cols = _csv(args.label_columns)

    casts: Dict[str, str] = {}
    if args.cast:
        for kv in args.cast.split(","):
            k, v = kv.split("=")
            casts[k.strip()] = v.strip()

    loader = build_loader(
        files=files,
        columns=columns,
        list_columns=list_cols,
        label_columns=label_cols,
        casts=casts,
        batch_size=args.batch_size,
        window_rows=args.window_rows,
        shuffle=args.shuffle,
        part_shuffle=args.part_shuffle,
        drop_last=args.drop_last,
        seed=args.seed,
        device_id=args.device_id,
        rmm_pool=args.rmm_pool,
    )

    if args.bench:
        run_bench(loader, args.bench)
        try:
            loader.dataset.close()
        except Exception:
            pass
        return

    _ = next(iter(loader))
    for b in tqdm(loader):
        feats, labels = _unpack(b)
        print(feats["X"].shape)

    try:
        loader.dataset.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()