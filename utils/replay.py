"""
Dummy replay to stream saved landmark frames at a target FPS.
"""

import json
import time
from typing import Any, Dict, List, Optional


class DummyReplay:
    def __init__(self, filepath: str, target_fps: int = 30) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            self.frames: List[Dict[str, Any]] = json.load(f)
        self.index = 0
        self.delay = 1.0 / target_fps if target_fps > 0 else 0.0

    def get_next(self) -> Optional[Dict[str, Any]]:
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        if self.delay:
            time.sleep(self.delay)
        return frame
