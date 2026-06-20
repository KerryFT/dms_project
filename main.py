"""
main.py — Entry point cho DMS project.

CÁCH DÙNG:
-----------

# 1. Smoke test (chạy local, không cần dataset thật):
python main.py smoke-test

# 2. Preprocess dataset UTA-URDD (chạy trên Kaggle hoặc local có MediaPipe):
python main.py preprocess \\
    --dataset-root /kaggle/input/uta-urdd-clahe-and-mesh \\
    --output-root  /kaggle/working/dms_windows \\
    --window-size  90 \\
    --stride       45 \\
    --fps          30

# 3. Train (chạy trên Kaggle sau khi preprocess xong):
python main.py train \\
    --data-root    /kaggle/working/dms_windows \\
    --checkpoint   /kaggle/working/checkpoints/best_model.pt \\
    --epochs       30 \\
    --batch-size   16 \\
    --use-residual

# 4. Evaluate checkpoint:
python main.py evaluate \\
    --data-root    /kaggle/working/dms_windows \\
    --checkpoint   /kaggle/working/checkpoints/best_model.pt \\
    --use-residual
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_imports():
    missing = []
    for pkg in ["torch", "torchvision", "cv2", "numpy", "sklearn", "xgboost"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Thiếu package: {missing}")
        print("Chạy: pip install -r requirements.txt")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mode 1: smoke-test
# ---------------------------------------------------------------------------

def run_smoke_test(args):
    print("=" * 60)
    print("SMOKE TEST — dữ liệu giả lập, không cần dataset thật")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Mode 2: preprocess
# ---------------------------------------------------------------------------

def run_preprocess(args):
    from src.data.uta_urdd import scan_dataset, assign_splits, preprocess_split

    print("=" * 60)
    print("PREPROCESS — UTA-URDD -> window .pt files")
    print("=" * 60)
    print(f"Dataset root : {args.dataset_root}")
    print(f"Output root  : {args.output_root}")
    print(f"Window size  : {args.window_size} frames")
    print(f"Stride       : {args.stride} frames")
    print(f"FPS          : {args.fps}")
    print(f"Frame subdir : {args.frame_subdir}")

    all_samples = scan_dataset(
        args.dataset_root,
        frame_subdir=args.frame_subdir,
        exclude_low_vigilant=not args.include_low_vigilant,
    )

    split_config = None
    if args.train_participants or args.val_participants or args.test_participants:
        split_config = {
            "train": args.train_participants or [],
            "val":   args.val_participants   or [],
            "test":  args.test_participants  or [],
        }

    splits = assign_splits(all_samples, split_config)

    for split_name, split_samples in splits.items():
        if not split_samples:
            print(f"SKIP {split_name}: không có sample.")
            continue
        out_dir = os.path.join(args.output_root, split_name)
        print(f"\n--- Preprocessing {split_name} ({len(split_samples)} sequences) -> {out_dir}")
        preprocess_split(
            split_samples,
            output_dir=out_dir,
            window_size=args.window_size,
            stride=args.stride,
            patch_size=args.patch_size,
            fps=args.fps,
        )

    print("\nPreprocess hoàn tất.")
    print("Bước tiếp theo: python main.py train --data-root", args.output_root)


# ---------------------------------------------------------------------------
# Mode 3: train
# ---------------------------------------------------------------------------

def run_train(args):
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import DMSWindowDataset, collate_windows
    from src.models.dms_model import DMSModel
    from src.training.losses import compute_class_weights_from_counts
    from src.training.train_loop import (
        precompute_xgb_oof, compute_xgb_proba_for_set,
        evaluate_xgb_only, train, TrainConfig,
    )

    print("=" * 60)
    print("TRAIN — Joint training: Stage 2+3+4+5")
    print("=" * 60)

    train_dir = os.path.join(args.data_root, "train")
    val_dir   = os.path.join(args.data_root, "val")
    for d in [train_dir, val_dir]:
        if not os.path.isdir(d):
            print(f"[ERROR] Không tìm thấy {d}. Chạy preprocess trước.")
            sys.exit(1)

    # Đường dẫn lưu XGBoost final model (song song với checkpoint DL).
    xgb_model_path = args.checkpoint.replace(".pt", "_xgb.pkl")

    # ----------------------------------------------------------------
    # Step 1: XGBoost
    # ----------------------------------------------------------------
    print("\n[Step 1/3] XGBoost OOF trên train set + lưu final model...")
    baseline = precompute_xgb_oof(
        train_dir,
        n_splits=args.xgb_folds,
        save_model_path=xgb_model_path,   # ← lưu để evaluate dùng lại
    )

    if args.use_residual:
        # Val set: dùng FINAL model (không dùng OOF — val chưa train trên đó).
        print("[Step 1/3] Tính XGBoost proba cho val set (final model)...")
        compute_xgb_proba_for_set(val_dir, xgb_model_path)

    # In baseline F1 trên val ngay để có con số so sánh.
    xgb_val_metrics = evaluate_xgb_only(val_dir, xgb_model_path)
    print(f"\n  XGBoost-only baseline (val):  "
          f"F1={xgb_val_metrics['f1_drowsy']:.4f}  "
          f"Recall={xgb_val_metrics['recall_drowsy']:.4f}  "
          f"Acc={xgb_val_metrics['accuracy']:.4f}")

    # ----------------------------------------------------------------
    # Step 2: DataLoader
    # ----------------------------------------------------------------
    print("\n[Step 2/3] Building DataLoaders...")
    train_ds = DMSWindowDataset(train_dir, require_xgb_oof=args.use_residual)
    val_ds   = DMSWindowDataset(val_dir,   require_xgb_oof=args.use_residual)

    n_workers = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_windows, num_workers=n_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_windows, num_workers=n_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} windows")

    # ----------------------------------------------------------------
    # Step 3: Train DL
    # ----------------------------------------------------------------
    print("\n[Step 3/3] Training DL pipeline (Stage 2→3→4→5)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels_list = [torch.load(p, weights_only=True)["label"].item()
                   for p in train_ds.sample_paths]
    n_alert, n_drowsy = labels_list.count(0), labels_list.count(1)
    class_weights = compute_class_weights_from_counts([max(n_alert, 1), max(n_drowsy, 1)])
    print(f"Label dist — Alert: {n_alert}  Drowsy: {n_drowsy} | weights: {class_weights.tolist()}")

    model = DMSModel(
        geometry_dim=6, film_hidden_dim=32, gru_hidden_dim=128,
        embed_dim=64, num_classes=2, pretrained_backbone=not args.no_pretrained,
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    config = TrainConfig(
        lambda_triplet=args.lambda_triplet,
        lambda_residual=args.lambda_residual,
        use_residual=args.use_residual,
        lr=args.lr, weight_decay=args.weight_decay,
    )

    history = train(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, class_weights=class_weights,
        config=config, checkpoint_path=args.checkpoint, verbose=True,
    )

    best_f1 = max(history["val_f1_drowsy"])
    print(f"\n{'='*60}")
    print(f"Training hoàn tất!")
    print(f"  XGBoost-only baseline val F1 : {xgb_val_metrics['f1_drowsy']:.4f}")
    print(f"  Full pipeline best val F1    : {best_f1:.4f}  "
          f"({'↑ tốt hơn' if best_f1 > xgb_val_metrics['f1_drowsy'] else '↓ cần điều chỉnh'})")
    print(f"  DL checkpoint : {args.checkpoint}")
    print(f"  XGBoost model : {xgb_model_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Mode 4: evaluate
# ---------------------------------------------------------------------------

def run_evaluate(args):
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import DMSWindowDataset, collate_windows
    from src.models.dms_model import DMSModel
    from src.training.train_loop import (
        evaluate, evaluate_xgb_only,
        compute_xgb_proba_for_set, load_checkpoint,
    )

    print("=" * 60)
    print("EVALUATE — So sánh 3 chế độ trên test set")
    print("=" * 60)

    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        print(f"[ERROR] Không tìm thấy {test_dir}.")
        sys.exit(1)

    # XGBoost model phải tồn tại song song với DL checkpoint.
    xgb_model_path = args.checkpoint.replace(".pt", "_xgb.pkl")
    if not os.path.exists(xgb_model_path):
        print(f"[ERROR] Không tìm thấy XGBoost model tại {xgb_model_path}.")
        print("Đảm bảo bạn đã chạy 'main.py train' để sinh ra file này.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # Bước 1: Tính XGBoost proba cho test set (dùng final model, không OOF)
    # ----------------------------------------------------------------
    print(f"\n[1/4] Tính XGBoost proba cho test set (final model từ {xgb_model_path})...")
    compute_xgb_proba_for_set(test_dir, xgb_model_path)

    # ----------------------------------------------------------------
    # Bước 2: XGBoost-only baseline
    # ----------------------------------------------------------------
    print("\n[2/4] Đánh giá XGBoost-only (baseline)...")
    xgb_metrics = evaluate_xgb_only(test_dir, xgb_model_path)

    # ----------------------------------------------------------------
    # Bước 3: Load DL model
    # ----------------------------------------------------------------
    print("\n[3/4] Load DL model và đánh giá...")
    model = DMSModel(pretrained_backbone=False)
    optimizer = torch.optim.Adam(model.parameters())
    epoch = load_checkpoint(model, optimizer, args.checkpoint)
    model.to(device)           # ← load_checkpoint dùng map_location="cpu", phải move sang device sau
    model.eval()
    print(f"  DL Checkpoint: epoch {epoch + 1}, file: {args.checkpoint}")

    test_ds     = DMSWindowDataset(test_dir, require_xgb_oof=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_windows)

    # Stage 3 only (không residual)
    dl_only_metrics  = evaluate(model, test_loader, device, use_residual=False)
    # Full pipeline (Stage 3 + Stage 5 residual)
    full_metrics     = evaluate(model, test_loader, device, use_residual=True)

    # ----------------------------------------------------------------
    # Bước 4: In bảng so sánh 3 chế độ
    # ----------------------------------------------------------------
    print("\n[4/4] Kết quả so sánh:")
    print(f"\n{'─'*65}")
    print(f"{'Chế độ':<30} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7}")
    print(f"{'─'*65}")
    for name, m in [
        ("XGBoost-only  (Stage 5a, baseline)", xgb_metrics),
        ("DL-only        (Stage 2+3+4)",        dl_only_metrics),
        ("Full pipeline  (Stage 2+3+4+5b)",      full_metrics),
    ]:
        print(
            f"{name:<30}  "
            f"{m['accuracy']:>6.4f}  "
            f"{m['precision_drowsy']:>6.4f}  "
            f"{m['recall_drowsy']:>7.4f}  "
            f"{m['f1_drowsy']:>6.4f}"
        )
    print(f"{'─'*65}")

    # Đánh giá mức độ cải thiện
    delta_dl   = full_metrics['f1_drowsy'] - xgb_metrics['f1_drowsy']
    delta_sign = "+" if delta_dl >= 0 else ""
    print(f"\n  ΔF1 (Full pipeline vs XGBoost baseline): {delta_sign}{delta_dl:.4f}")
    if delta_dl >= 0.01:
        print("  ✓ DL pipeline cải thiện rõ rệt so với baseline.")
    elif delta_dl >= 0:
        print("  ~ DL pipeline cải thiện nhẹ. Thử tăng epoch hoặc điều chỉnh lambda_residual.")
    else:
        print("  ✗ DL pipeline chưa vượt baseline. Xem gợi ý trong README — Ablation study.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMS Drowsiness Detection Pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- smoke-test ---
    sub.add_parser("smoke-test", help="Chạy toàn bộ pytest với dữ liệu giả lập (local)")

    # --- preprocess ---
    p_pre = sub.add_parser("preprocess", help="Preprocess UTA-URDD -> .pt window files (Kaggle)")
    p_pre.add_argument("--dataset-root",      required=True,  help="Root của UTA-URDD dataset")
    p_pre.add_argument("--output-root",        required=True,  help="Nơi lưu .pt files")
    p_pre.add_argument("--frame-subdir",       default="frames_clahe")
    p_pre.add_argument("--window-size",        type=int,   default=90)
    p_pre.add_argument("--stride",             type=int,   default=45)
    p_pre.add_argument("--patch-size",         type=int,   default=64)
    p_pre.add_argument("--fps",                type=float, default=30.0)
    p_pre.add_argument("--include-low-vigilant", action="store_true",
                       help="Include mức '5' (Low Vigilant) như label 2 cho triplet 3-class")
    p_pre.add_argument("--train-participants", nargs="+",
                       default=["participant1","participant2","participant3","participant4"])
    p_pre.add_argument("--val-participants",   nargs="+", default=["participant5"])
    p_pre.add_argument("--test-participants",  nargs="+", default=["participant6"])

    # --- train ---
    p_tr = sub.add_parser("train", help="Train end-to-end (Kaggle, cần GPU)")
    p_tr.add_argument("--data-root",       required=True)
    p_tr.add_argument("--checkpoint",      default="checkpoints/best_model.pt")
    p_tr.add_argument("--epochs",          type=int,   default=30)
    p_tr.add_argument("--batch-size",      type=int,   default=16)
    p_tr.add_argument("--lr",              type=float, default=1e-3)
    p_tr.add_argument("--weight-decay",    type=float, default=1e-4)
    p_tr.add_argument("--lambda-triplet",  type=float, default=0.2)
    p_tr.add_argument("--lambda-residual", type=float, default=0.3)
    p_tr.add_argument("--xgb-folds",       type=int,   default=5)
    p_tr.add_argument("--use-residual",    action="store_true", default=True)
    p_tr.add_argument("--no-pretrained",   action="store_true",
                       help="Không load ImageNet pretrained weights (mặc định: load)")

    # --- evaluate ---
    p_ev = sub.add_parser("evaluate", help="Evaluate checkpoint trên test set")
    p_ev.add_argument("--data-root",    required=True)
    p_ev.add_argument("--checkpoint",   required=True)
    p_ev.add_argument("--use-residual", action="store_true", default=True)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _check_imports()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "smoke-test": run_smoke_test,
        "preprocess": run_preprocess,
        "train":      run_train,
        "evaluate":   run_evaluate,
    }
    dispatch[args.mode](args)
