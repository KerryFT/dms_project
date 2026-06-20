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
    from src.training.train_loop import precompute_xgb_oof, attach_xgb_proba_for_eval, train, TrainConfig

    print("=" * 60)
    print("TRAIN — Joint training: Stage 2+3+4+5")
    print("=" * 60)

    train_dir = os.path.join(args.data_root, "train")
    val_dir   = os.path.join(args.data_root, "val")
    for d in [train_dir, val_dir]:
        if not os.path.isdir(d):
            print(f"[ERROR] Không tìm thấy {d}. Chạy preprocess trước.")
            sys.exit(1)

    # Step 1: XGBoost OOF — CHỈ fit trên train (đúng spec: fit_oof tránh
    # leak cho train, fit_final là model "deployment" dùng để predict cho
    # các split còn lại).
    print("\n[Step 1/3] Precompute XGBoost OOF probabilities (train set)...")
    baseline = precompute_xgb_oof(train_dir, n_splits=args.xgb_folds)
    if args.use_residual:
        # Val KHÔNG fit OOF riêng (làm vậy sẽ fit model mới bằng label của
        # chính val — lệch khỏi spec). Dùng predict_proba() của model đã
        # fit trên train, giống hệt cách test set sẽ được xử lý lúc evaluate.
        print("[Step 1/3] Gắn XGBoost proba (predict từ model train) cho val set...")
        attach_xgb_proba_for_eval(val_dir, baseline)

    # Step 2: Dataloader.
    print("\n[Step 2/3] Building DataLoaders...")
    train_ds = DMSWindowDataset(train_dir, require_xgb_oof=args.use_residual)
    val_ds   = DMSWindowDataset(val_dir,   require_xgb_oof=args.use_residual)

    import torch
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

    # Step 3: Train.
    print("\n[Step 3/3] Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = [s["label"].item() for s in [torch.load(p) for p in train_ds.sample_paths]]
    n_alert   = labels.count(0)
    n_drowsy  = labels.count(1)
    class_weights = compute_class_weights_from_counts([max(n_alert, 1), max(n_drowsy, 1)])
    print(f"Class distribution — Alert: {n_alert}, Drowsy: {n_drowsy}")
    print(f"Class weights: {class_weights.tolist()}")

    model = DMSModel(
        geometry_dim=6,
        film_hidden_dim=32,
        gru_hidden_dim=128,
        embed_dim=64,
        num_classes=2,
        pretrained_backbone=not args.no_pretrained,
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    config = TrainConfig(
        lambda_triplet=args.lambda_triplet,
        lambda_residual=args.lambda_residual,
        use_residual=args.use_residual,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = train(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs,
        class_weights=class_weights,
        config=config,
        checkpoint_path=args.checkpoint,
        verbose=True,
    )

    best_f1 = max(history["val_f1_drowsy"])
    print(f"\nTrain hoàn tất. Best val F1 (Drowsy): {best_f1:.4f}")
    print(f"Checkpoint lưu tại: {args.checkpoint}")


# ---------------------------------------------------------------------------
# Mode 4: evaluate
# ---------------------------------------------------------------------------

def run_evaluate(args):
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import DMSWindowDataset, collate_windows
    from src.models.dms_model import DMSModel
    from src.training.train_loop import evaluate, load_checkpoint, precompute_xgb_oof, attach_xgb_proba_for_eval

    print("=" * 60)
    print("EVALUATE")
    print("=" * 60)

    train_dir = os.path.join(args.data_root, "train")
    test_dir  = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        print(f"[ERROR] Không tìm thấy {test_dir}.")
        sys.exit(1)

    if args.use_residual:
        if not os.path.isdir(train_dir):
            print(f"[ERROR] Cần {train_dir} để refit XGBoost baseline cho residual "
                  f"(model XGBoost lúc train không được lưu lại, nên evaluate() chạy "
                  f"ở process riêng phải refit lại trên train — KHÔNG dùng label test).")
            sys.exit(1)
        # Refit XGBoost trên train (rẻ, chỉ geometry + CPU, không liên quan
        # gì tới checkpoint neural net) để có model giống hệt lúc train,
        # rồi PREDICT (không fit) cho test. Test set không hề bị động tới.
        print("\n[Step 1/2] Refit XGBoost baseline trên train set...")
        baseline = precompute_xgb_oof(train_dir, n_splits=args.xgb_folds)
        print("[Step 1/2] Gắn XGBoost proba (predict từ model train) cho test set...")
        attach_xgb_proba_for_eval(test_dir, baseline)

    print("\n[Step 2/2] Load checkpoint & evaluate...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMSModel(pretrained_backbone=False)
    model.to(device)  # train() làm việc này nội bộ, nhưng evaluate() thì không — phải tự move ở đây
    optimizer = torch.optim.Adam(model.parameters())
    epoch = load_checkpoint(model, optimizer, args.checkpoint)
    print(f"Loaded checkpoint: epoch {epoch + 1}, file: {args.checkpoint}")

    test_ds = DMSWindowDataset(test_dir, require_xgb_oof=args.use_residual)
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_windows,
    )

    metrics = evaluate(model, test_loader, device, use_residual=args.use_residual)
    print("\n--- Test Set Results ---")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")

    print(f"\nBaseline F1 (XGBoost only): kiểm tra file {args.data_root}/xgb_baseline_f1.txt nếu đã lưu.")


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
    # LƯU Ý: tên thư mục THẬT trong dataset UTA-URDD bị lỗi chính tả ở
    # participant 2 và 4 ("partcipant2"/"partcipant4", thiếu chữ "i") —
    # xem uta_urdd.py::DEFAULT_SPLIT. Default ở đây PHẢI khớp với tên
    # thư mục thật, không phải chính tả "đúng chuẩn", nếu không
    # assign_splits() sẽ không nhận diện được participant 2/4 và tự
    # động (âm thầm) đẩy chúng vào train qua cơ chế auto-fix — có thể
    # gây leakage nếu participant đó đáng lẽ thuộc val/test.
    # => Cách fix triệt để hơn: đổi tên thư mục trên dataset (nếu có
    # quyền ghi), hoặc chuẩn hoá tên participant trong assign_splits()
    # trước khi so khớp thay vì hard-code chính tả lỗi ở đây.
    p_pre.add_argument("--train-participants", nargs="+",
                       default=["participant1","partcipant2","participant3","partcipant4"])
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
    # store_true + default=True khiến --use-residual LUÔN True dù có truyền
    # flag hay không (không có cách tắt qua CLI). Thêm --no-residual để
    # thực sự tắt được, dùng cho ablation "with vs without residual".
    p_tr.add_argument("--use-residual",    dest="use_residual", action="store_true", default=True)
    p_tr.add_argument("--no-residual",     dest="use_residual", action="store_false",
                       help="Tắt Stage 5 residual fallback (ablation)")
    p_tr.add_argument("--no-pretrained",   action="store_true",
                       help="Không load ImageNet pretrained weights (mặc định: load)")

    # --- evaluate ---
    p_ev = sub.add_parser("evaluate", help="Evaluate checkpoint trên test set")
    p_ev.add_argument("--data-root",    required=True)
    p_ev.add_argument("--checkpoint",   required=True)
    p_ev.add_argument("--xgb-folds",    type=int, default=5,
                       help="Dùng để refit XGBoost baseline trên train set (xem run_evaluate)")
    p_ev.add_argument("--use-residual", dest="use_residual", action="store_true", default=True)
    p_ev.add_argument("--no-residual",  dest="use_residual", action="store_false",
                       help="Tắt Stage 5 residual fallback (ablation)")

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
