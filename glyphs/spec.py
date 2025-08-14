from dataclasses import dataclass
from typing import List, Dict, Any

# Board glyph channels (keep original 10 + new channels appended)
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
  "underdef_white",
  "underdef_black",
  "central_control_white",
  "central_control_black",
  "contact_threat_white",
  "contact_threat_black",
  "piece_highlight_white",
  "piece_highlight_black",
  "pawn_chain_white",
  "pawn_chain_black",
  "empty_square_highlight"
]

@dataclass
class GlyphMap:
    """
    Container for glyph logits or probabilities as C channels of 8x8.
    """
    tensor: "np.ndarray"  # shape [C,8,8], float

    def to_serializable(self) -> Dict[str, Any]:
        import numpy as np
        out = {}
        for idx, name in enumerate(GLYPH_CHANNELS):
            out[name] = self.tensor[idx].tolist()
        return out
