"""
Step 3: Stable webcam capture (CAP_DSHOW) + MediaPipe Tasks Hands (21 landmarks per hand).
Outputs per-frame hand landmarks and draws them for debug.
"""

import json
import pathlib
import time
import urllib.request
import zipfile
from typing import Dict, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.smoothing import LandmarkSmoother
from utils.recorder import LandmarkRecorder
from utils.replay import DummyReplay


USE_DUMMY = False
DUMMY_FILE = "data/raw_landmarks/sample_sign_sequence.json"
RECORD_FILENAME = f"session_{int(time.time())}.json"


MODELS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def _ensure_model(path: pathlib.Path, url: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _valid_zip(p: pathlib.Path) -> bool:
        try:
            with zipfile.ZipFile(p, "r") as zf:
                return zf.testzip() is None
        except Exception:
            return False

    if path.exists() and _valid_zip(path):
        return path

    tmp_path = path.with_suffix(path.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    urllib.request.urlretrieve(url, tmp_path)
    if not _valid_zip(tmp_path):
        raise RuntimeError(f"Failed to fetch valid model from {url}")
    tmp_path.replace(path)
    return path


def _capture_preview(cap: cv2.VideoCapture, target_fps: int) -> int:
    frame_interval = 1.0 / target_fps
    prev_time = 0.0
    last_frame_time = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
        prev_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = rgb_frame  # placeholder to emphasize conversion occurs

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("EchoConnect - Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        now = time.perf_counter()
        elapsed = now - last_frame_time
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        last_frame_time = time.perf_counter()

    return 0


def _run_dummy() -> int:
    replay = DummyReplay(DUMMY_FILE, target_fps=30)
    for frame in iter(replay.get_next, None):
        # Hook: send frame downstream; here we just print ids.
        print(f"Replay frame {frame['frame_id']}")
    return 0


def _extract_hands(result: vision.HandLandmarkerResult) -> List[Dict]:
    hands: List[Dict] = []
    if not result or not result.hand_landmarks:
        return hands

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in hand_landmarks
        ]
        hands.append({"hand_id": label, "landmarks": landmarks})
    return hands


def main() -> int:
    if USE_DUMMY:
        return _run_dummy()

    target_fps = 30
    frame_interval = 1.0 / target_fps
    prev_time = 0.0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Webcam not available.")
        return 1

    models_dir = pathlib.Path("data/models")
    try:
        hand_model_path = _ensure_model(models_dir / "hand_landmarker.task", MODELS["hand"])
        face_model_path = _ensure_model(models_dir / "face_landmarker.task", MODELS["face"])
    except Exception as exc:  # fall back to capture-only if download fails
        print(f"Model download failed: {exc}\nRunning capture-only preview for Step 2.")
        result = _capture_preview(cap, target_fps)
        cap.release()
        cv2.destroyAllWindows()
        return result

    # Save a dummy sample payload for downstream consumers.
    samples_dir = pathlib.Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_payload = {
        "frame_id": 0,
        "timestamp": 0.0,
        "hands": [
            {
                "hand_id": "Right",
                "missing": False,
                "landmarks": [[0.5, 0.5, 0.0] for _ in range(21)],
            }
        ],
        "faces": [
            {
                "face_id": "face0",
                "missing": False,
                "landmarks": [[0.4, 0.4, 0.0] for _ in range(468)],
            }
        ],
    }
    with open(samples_dir / "landmarks_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_payload, f)

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    start_time = time.perf_counter()
    last_frame_time = start_time
    frame_id = 0
    smoother_hands = LandmarkSmoother()
    smoother_face = LandmarkSmoother()
    recorder = LandmarkRecorder()

    with vision.HandLandmarker.create_from_options(hand_options) as hands_model, vision.FaceLandmarker.create_from_options(face_options) as face_model:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read frame.")
                break

            # FPS measurement (real FPS from loop timings)
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time

            # MediaPipe expects RGB input; mark as not writeable for perf.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            hand_result = hands_model.detect_for_video(mp_image, timestamp_ms)
            face_result = face_model.detect_for_video(mp_image, timestamp_ms)

            hands = _extract_hands(hand_result)
            detected_labels = set()

            smoothed_hands = []
            for hand in hands:
                label = hand["hand_id"]
                detected_labels.add(label)
                raw_landmarks = [
                    [lm["x"], lm["y"], lm["z"]] for lm in hand["landmarks"]
                ]
                smoothed, missing_flag = smoother_hands.smooth(label, raw_landmarks)
                smoothed_hands.append(
                    {
                        "hand_id": label,
                        "missing": missing_flag,
                        "landmarks": smoothed,
                    }
                )

            for label in smoother_hands.known_labels():
                if label in detected_labels:
                    continue
                smoothed, missing_flag = smoother_hands.handle_missing(label)
                if smoothed is None:
                    continue
                smoothed_hands.append(
                    {
                        "hand_id": label,
                        "missing": missing_flag,
                        "landmarks": smoothed,
                    }
                )

            smoothed_faces = []
            if face_result and face_result.face_landmarks:
                raw_face = [
                    [lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0][:468]
                ]
                smoothed, missing_flag = smoother_face.smooth("face0", raw_face)
                smoothed_faces.append(
                    {
                        "face_id": "face0",
                        "missing": missing_flag,
                        "landmarks": smoothed,
                    }
                )
            else:
                smoothed, missing_flag = smoother_face.handle_missing("face0")
                if smoothed is not None:
                    smoothed_faces.append(
                        {
                            "face_id": "face0",
                            "missing": missing_flag,
                            "landmarks": smoothed,
                        }
                    )

            # Draw simple landmark dots for preview to avoid dependency on solutions drawing utils.
            h, w, _ = frame.shape
            for hand in smoothed_hands:
                if hand["missing"]:
                    continue
                for lm in hand["landmarks"]:
                    cx, cy = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            for face in smoothed_faces:
                if face["missing"]:
                    continue
                for lm in face["landmarks"]:
                    cx, cy = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 128, 255), -1)

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            hands_dict = {hand["hand_id"]: hand["landmarks"] for hand in smoothed_hands if not hand["missing"]}
            face_list = smoothed_faces[0]["landmarks"] if smoothed_faces else []
            missing_hands_flag = len(hands_dict) == 0
            missing_face_flag = not smoothed_faces or smoothed_faces[0]["missing"]

            recorder.add_frame(
                frame_id=frame_id,
                hands=hands_dict,
                face=face_list,
                fps=int(fps),
                missing_hands=missing_hands_flag,
                missing_face=missing_face_flag,
            )

            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break

            now = time.perf_counter()
            elapsed = now - last_frame_time
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            last_frame_time = time.perf_counter()
            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    recorder.save(RECORD_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
