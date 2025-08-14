from dataclasses import dataclass
from typing import List, Dict, Any

GLYPH_CHANNELS = [
    "attack_white",
    "attack_black",
    "defended_white",
    "defended_black",
    "hanging_white",
    "hanging_black",
    "pinned_white",
    "pinned_black",
    "king_danger_white",
    "king_danger_black",
]

@dataclass
class GlyphMap:
    """
    Container for glyph logits or probabilities as 10 channels of 8x8.
    """
    tensor: "np.ndarray"  # shape [10,8,8], float

    def to_serializable(self) -> Dict[str, Any]:
        import numpy as np
        out = {}
        for idx, name in enumerate(GLYPH_CHANNELS):
            out[name] = (self.tensor[idx] > 0.5).astype(int).tolist()
        return out
