"""
Landmark smoothing utilities for hand stabilization (Step 4).
Applies missing-frame reuse, exponential moving average, and outlier clamp.
"""

from typing import Dict, List, Tuple

import numpy as np


ALPHA = 0.7
MAX_DELTA = 0.1
MAX_MISSING_FRAMES = 3


class LandmarkSmoother:
	def __init__(self) -> None:
		self.prev: Dict[str, np.ndarray] = {}
		self.missing_count: Dict[str, int] = {}

	def smooth(self, hand_label: str, current_landmarks: List[List[float]]) -> Tuple[List[List[float]], bool]:
		current = np.array(current_landmarks, dtype=float)

		if hand_label not in self.prev:
			self.prev[hand_label] = current
			self.missing_count[hand_label] = 0
			return current.tolist(), False

		smooth = ALPHA * current + (1.0 - ALPHA) * self.prev[hand_label]

		delta = smooth - self.prev[hand_label]
		delta = np.clip(delta, -MAX_DELTA, MAX_DELTA)
		smooth = self.prev[hand_label] + delta

		self.prev[hand_label] = smooth
		self.missing_count[hand_label] = 0
		return smooth.tolist(), False

	def handle_missing(self, hand_label: str) -> Tuple[List[List[float]] | None, bool]:
		if hand_label not in self.prev:
			return None, True

		self.missing_count[hand_label] = self.missing_count.get(hand_label, 0) + 1

		if self.missing_count[hand_label] > MAX_MISSING_FRAMES:
			del self.prev[hand_label]
			del self.missing_count[hand_label]
			return None, True

		return self.prev[hand_label].tolist(), True

	def known_labels(self) -> List[str]:
		return list(self.prev.keys())

