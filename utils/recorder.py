"""
Landmark recorder for saving synchronized hand and face streams.
"""

import json
import os
import time
from typing import Any, Dict, List


class LandmarkRecorder:
    def __init__(self, save_dir: str = "data/raw_landmarks") -> None:
        self.frames: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def add_frame(
        self,
        frame_id: int,
        hands: Dict[str, List[List[float]]],
        face: List[List[float]],
        fps: int,
        missing_hands: bool,
        missing_face: bool,
    ) -> None:
        frame_data = {
            "frame_id": frame_id,
            "timestamp": round(time.time() - self.start_time, 3),
            "hands": hands,
            "face": face,
            "fps": fps,
            "missing": {
                "hands": missing_hands,
                "face": missing_face,
            },
        }
        self.frames.append(frame_data)

    def save(self, filename: str) -> str:
        path = os.path.join(self.save_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.frames, f, indent=2)
        print(f"[✓] Landmark data saved to {path}")
        return path
