"""
UTA-URDD Dataset Adapter (src/data/uta_urdd.py)

Đọc cấu trúc thư mục UTA-URDD và sinh window .pt files cho pipeline.

Cấu trúc dataset:
    frames_clahe/
        0/          <- buồn ngủ (Drowsy)
            participant1/ <- ảnh từng frame (đã CLAHE)
            participant2/
            ...
        5/          <- gần buồn ngủ (Low Vigilant)
            participant1/
            ...
        10/         <- thức (Alert)
            participant1/
            ...
    frames_mesh/    <- không dùng để train (chỉ visualization)
    csv/            <- metadata (optional)
    failed_detections/

QUAN TRỌNG — Label mapping:
    "10" (Alert)         -> 0
    "0"  (Drowsy)        -> 1
    "5"  (Low Vigilant)  -> 2  (dùng cho triplet 3-class, hoặc loại bỏ theo config)

QUAN TRỌNG — Split theo participant (KHÔNG split ngẫu nhiên theo frame):
    Nếu split theo frame, các frame liên tiếp từ cùng 1 người có thể vào
    cả train và val, khiến model "thuộc mặt" participant và F1 trên val
    bị inflate — đây là data leakage điển hình trong face/driver dataset.
    Ở đây mình split TOÀN BỘ participant vào đúng 1 partition (train/val/test).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LEVEL_TO_LABEL: Dict[str, int] = {
    "10": 0,  # Alert
    "0":  1,  # Drowsy
    "5":  2,  # Low Vigilant (tuỳ config: dùng cho triplet hoặc exclude)
}

LABEL_NAMES = {0: "Alert", 1: "Drowsy", 2: "LowVigilant"}

# Khi chạy ở chế độ binary, loại bỏ class 5 (Low Vigilant) khỏi training
# classification nhưng vẫn có thể dùng cho triplet mining.
# Nếu muốn include class 5 vào binary (merge với Drowsy), đổi thành:
#   BINARY_MERGE_LOWVIGILANT_AS = 1
BINARY_EXCLUDE_LOWVIGILANT = True

# ---------------------------------------------------------------------------
# Participant split (mặc định; đổi tuỳ dataset size và số participant)
# ---------------------------------------------------------------------------
DEFAULT_SPLIT = {
    "train": ["participant1", "partcipant2", "participant3", "partcipant4"],
    "val":   ["participant5"],
    "test":  ["participant6"],
}


# ---------------------------------------------------------------------------
# Frame scanning
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _scan_participant_frames(part_dir: Path) -> List[Path]:
    """Lấy danh sách frame theo thứ tự tên file (quan trọng: thứ tự = thời gian)."""
    frames = sorted(
        [p for p in part_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS],
        key=lambda p: p.stem,
    )
    return frames


def scan_dataset(
    dataset_root: str,
    frame_subdir: str = "frames_clahe",
    exclude_low_vigilant: bool = BINARY_EXCLUDE_LOWVIGILANT,
) -> List[Dict]:
    """
    Quét toàn bộ dataset và trả về danh sách sample info.

    Returns:
        List of dicts, mỗi dict chứa:
            frame_paths: List[Path]  (toàn bộ frame của participant, đã sort)
            level:       str         ("0", "5", "10")
            label:       int         (0, 1, hoặc 2)
            participant: str         ("participant1", ...)
    """
    root = Path(dataset_root) / frame_subdir
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy {root}. Kiểm tra dataset_root và frame_subdir.")

    samples = []
    for level_dir in sorted(root.iterdir()):
        if not level_dir.is_dir() or level_dir.name not in LEVEL_TO_LABEL:
            continue
        label = LEVEL_TO_LABEL[level_dir.name]
        if exclude_low_vigilant and label == 2:
            continue

        for part_dir in sorted(level_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            frames = _scan_participant_frames(part_dir)
            if len(frames) == 0:
                continue
            samples.append({
                "frame_paths": frames,
                "level": level_dir.name,
                "label": label,
                "participant": part_dir.name,
            })

    print(f"Scanned {len(samples)} participant-level entries from {root}")
    return samples

def assign_splits(samples: List[Dict], split_config: Dict[str, List[str]] = None) -> Dict[str, List[Dict]]:
    """Gán mỗi sample vào train/val/test dựa trên participant ID.
    Tự động thêm các participant thiếu vào tập train để không bị mất dữ liệu.
    """
    # Tạo bản sao của config để tránh ghi đè lên biến toàn cục
    if split_config is None:
        try:
            current_config = {k: list(v) for k, v in DEFAULT_SPLIT.items()}
        except NameError:
            current_config = {"train": [], "val": [], "test": []}
    else:
        current_config = {k: list(v) for k, v in split_config.items()}

    result = {"train": [], "val": [], "test": []}
    auto_assigned = []

    for s in samples:
        p = s["participant"]
        assigned = False
        
        # Kiểm tra xem participant đã nằm trong split nào chưa
        for split_name, participants in current_config.items():
            if p in participants:
                result[split_name].append(s)
                assigned = True
                break
                
        # Nếu KHÔNG CÓ trong config, tự động đẩy vào tập 'train' để xử lý luôn
        if not assigned:
            current_config["train"].append(p)
            result["train"].append(s)
            auto_assigned.append(p)

    # In thông báo để bạn theo dõi xem có ai bị nhận diện sai không
    if auto_assigned:
        print(f"\n💡 TỰ ĐỘNG FIX CONFIG: Phát hiện {len(set(auto_assigned))} participant nằm ngoài danh sách: {set(auto_assigned)}")
        print("   -> Đã tự động xếp họ vào tập 'train' để đảm bảo không bị SKIP dữ liệu.\n")

    # In thống kê kết quả phân chia tập dữ liệu
    for split_name, items in result.items():
        if items:
            # Lấy danh sách label thực tế xuất hiện trong split này
            existing_labels = set(x['label'] for x in items)
            label_stats = {LABEL_NAMES[i]: sum(1 for x in items if x['label'] == i) for i in existing_labels}
            print(f"  {split_name}: {len(items)} sequences | labels: {label_stats}")
        else:
            print(f"  {split_name}: 0 sequences")
            
    return result
'''
def assign_splits(samples: List[Dict], split_config: Dict[str, List[str]] = None) -> Dict[str, List[Dict]]:
    """Gán mỗi sample vào train/val/test dựa trên participant ID."""
    if split_config is None:
        split_config = DEFAULT_SPLIT
    all_defined = set(p for v in split_config.values() for p in v)
    result = {"train": [], "val": [], "test": []}
    skipped = []
    for s in samples:
        p = s["participant"]
        assigned = False
        for split_name, participants in split_config.items():
            if p in participants:
                result[split_name].append(s)
                assigned = True
                break
        if not assigned:
            skipped.append(p)
    if skipped:
        print(f"CẢNH BÁO: {len(set(skipped))} participant không có trong split_config: {set(skipped)}")
        print("Thêm vào split_config hoặc họ sẽ bị bỏ qua khi preprocess.")
    for split_name, items in result.items():
        print(f"  {split_name}: {len(items)} sequences | labels: "
              f"{ {LABEL_NAMES[i]: sum(1 for x in items if x['label']==i) for i in set(x['label'] for x in items)} }")
    return result
'''

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sliding_windows(
    frame_paths: List[Path],
    window_size: int,
    stride: int,
) -> List[List[Path]]:
    """Tạo danh sách các window (list of frame paths) từ 1 sequence."""
    windows = []
    n = len(frame_paths)
    start = 0
    while start + window_size <= n:
        windows.append(frame_paths[start: start + window_size])
        start += stride
    # Nếu còn dư (đuôi sequence), lấy window cuối thẳng về end
    if start < n and n - start >= window_size // 2:
        windows.append(frame_paths[n - window_size: n])
    return windows


# ---------------------------------------------------------------------------
# Preprocess: chạy MediaPipe + Stage 1 + lưu .pt file
# ---------------------------------------------------------------------------

def preprocess_split(
    split_samples: List[Dict],
    output_dir: str,
    window_size: int = 90,
    stride: int = 45,
    patch_size: int = 64,
    fps: float = 30.0,
    confidence_threshold: float = 0.5,
    mp_min_detection_confidence: float = 0.3,
) -> List[str]:
    """
    Chạy MediaPipe FaceMesh trên từng frame, trích xuất Stage 1 features,
    lưu window .pt files vào output_dir.

    Args:
        split_samples:  output của assign_splits() cho một partition.
        output_dir:     nơi lưu .pt files (mỗi window = 1 file).
        window_size:    số frame mỗi window (90 frame @ 30fps = 3 giây).
        stride:         bước trượt window (45 = overlap 50%).
        fps:            FPS của video gốc (dùng cho rolling normalizer).
        confidence_threshold: ngưỡng MediaPipe presence score để gating.
    Returns:
        List đường dẫn các .pt file đã lưu.
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "MediaPipe chưa được cài. Chạy: pip install mediapipe\n"
            "Lưu ý: bước preprocess này cần chạy trên máy có MediaPipe,\n"
            "KHÔNG thể chạy trong sandbox hiện tại (không có mediapipe).\n"
            "Chạy trên Kaggle hoặc máy local của bạn với dataset thật."
        )

    from ..features.geometry import GeometryFeatureExtractor
    from ..features.patches import EyePatchExtractor
    from ..data.preprocessing import build_window_sample, save_window_sample

    os.makedirs(output_dir, exist_ok=True)
    mp_face_mesh = mp.solutions.face_mesh

    all_paths = []
    total_windows = 0
    failed_frames = 0

    for sample in tqdm(split_samples, desc=f"Preprocessing -> {output_dir}"):
        frame_paths = sample["frame_paths"]
        label = sample["label"]
        participant = sample["participant"]
        level = sample["level"]

        windows = sliding_windows(frame_paths, window_size, stride)

        # Một GeometryFeatureExtractor per sequence (giữ state rolling normalizer
        # liên tục qua toàn bộ sequence; reset khi sang participant mới).
        geom_extractor = GeometryFeatureExtractor(
            frame_width=640, frame_height=480,
            window_seconds=10.0, fps=fps,
        )
        patch_extractor = EyePatchExtractor(target_size=patch_size)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=mp_min_detection_confidence,
        ) as face_mesh:
            for win_idx, window_paths in enumerate(windows):
                frames, landmarks_list, confidences = [], [], []
                window_failed = False

                for fpath in window_paths:
                    img = cv2.imread(str(fpath))
                    if img is None:
                        failed_frames += 1
                        window_failed = True
                        break
                    h, w = img.shape[:2]

                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = face_mesh.process(rgb)

                    if result.multi_face_landmarks:
                        lm = result.multi_face_landmarks[0]
                        landmarks_px = {
                            i: (lm.landmark[i].x * w, lm.landmark[i].y * h)
                            for i in range(len(lm.landmark))
                        }
                        # Dùng presence score MediaPipe làm confidence nếu có
                        conf = float(getattr(lm.landmark[0], "presence", 0.9))
                        conf = max(conf, confidence_threshold)  # floor cho detected frames
                    else:
                        # Không detect được mặt: dùng dict rỗng, confidence thấp
                        landmarks_px = {}
                        conf = 0.0
                        failed_frames += 1

                    frames.append(img)
                    landmarks_list.append(landmarks_px)
                    confidences.append(conf)

                if window_failed:
                    continue

                out_path = os.path.join(
                    output_dir,
                    f"{participant}_lvl{level}_w{win_idx:04d}.pt",
                )
                try:
                    sample_pt = build_window_sample(
                        frames, landmarks_list, confidences, label,
                        geom_extractor, patch_extractor,
                    )
                    save_window_sample(sample_pt, out_path)
                    all_paths.append(out_path)
                    total_windows += 1
                except Exception as e:
                    print(f"  SKIP {out_path}: {e}")

    print(f"Finished: {total_windows} windows saved, {failed_frames} frames failed detection.")
    return all_paths


# ---------------------------------------------------------------------------
# Triplet mining với 3 lớp (BÂY GIỜ CÓ THỂ DÙNG ĐÚNG SPEC GỐC)
# ---------------------------------------------------------------------------
"""
GHI CHÚ: Với dataset UTA-URDD có 3 mức 0/5/10, giờ có thể implement đúng
triplet spec ban đầu:
    Anchor   = Low Vigilant (label 2, folder "5")
    Positive = Drowsy       (label 1, folder "0")
    Negative = Alert        (label 0, folder "10")

Để dùng, trong src/training/triplet.py, thay đổi semi_hard_triplet_indices
để nhận anchor_class, positive_class, negative_class:

    DROWSY_AWARE_TRIPLET = {
        "anchor_class":   2,   # Low Vigilant
        "positive_class": 1,   # Drowsy
        "negative_class": 0,   # Alert
    }

Hoặc giữ nguyên code triplet hiện tại (Drowsy vs Alert) và chỉ include
class 5 như một hard positive — tuỳ ablation bạn muốn chạy cho paper.
"""
