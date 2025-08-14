import numpy as np
from typing import Optional, Sequence
from glyphs.spec import GLYPH_CHANNELS

def composite_any(ch_tensor: np.ndarray,
                  include_names: Optional[Sequence[str]] = None,
                  thresh: float = 0.5) -> np.ndarray:
    """
    Reduce a [C,8,8] tensor to a [8,8] mask by per-cell OR over channels.
    - ch_tensor: np.ndarray [C,8,8] (float or bool)
    - include_names: subset of channel names to include; if None, include all
    - thresh: applied if array is float
    Returns: np.ndarray [8,8] with {0,1} values
    """
    if ch_tensor.ndim != 3 or ch_tensor.shape[1:] != (8, 8):
        raise ValueError(f"Expected [C,8,8], got {ch_tensor.shape}")
    if include_names is not None:
        idxs = [GLYPH_CHANNELS.index(n) for n in include_names if n in GLYPH_CHANNELS]
        if not idxs:
            return np.zeros((8, 8), dtype=np.float32)
        ch_tensor = ch_tensor[idxs]
    mask = (ch_tensor > thresh) if ch_tensor.dtype != bool else ch_tensor
    return mask.any(axis=0).astype(np.float32)
