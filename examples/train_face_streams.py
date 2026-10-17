"""Train the masked temporal face-stream classifier on official WLASL splits.

The validation split is used for early stopping. This example does not load or tune on test data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from kinetic_augment.evaluation.datasets.face_features import (
    WLASLFaceDataset,
    collate_face_samples,
)
from kinetic_augment.evaluation.models import FaceStreamClassifier, SLRTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root", type=Path, required=True, help="Face feature archive directory"
    )
    parser.add_argument(
        "--annotation-root", type=Path, required=True, help="Directory containing WLASL JSON"
    )
    parser.add_argument("--annotation-file", default="WLASL_v0.3.json")
    parser.add_argument(
        "--subset", choices=("wlasl100", "wlasl300", "wlasl1000", "wlasl2000"), default="wlasl100"
    )
    parser.add_argument("--face-space", choices=("image", "head_local"), default="image")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, help="Optional checkpoint path")
    parser.add_argument("--body-only", action="store_true", help="Run a body_hands-only ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    common = {
        "root_path": args.annotation_root,
        "feature_dir": args.feature_root,
        "annotation_file": args.annotation_file,
        "subset": args.subset,
        "face_space": args.face_space,
    }
    train_data = WLASLFaceDataset(split="train", **common)
    val_data = WLASLFaceDataset(split="val", **common)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_face_samples,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_face_samples,
    )

    if args.body_only:
        stream_dims = {"body_hands": 96}
    else:
        stream_dims = {
            "body_hands": 96,
            "face": 52 if args.face_space == "image" else 78,
            "head": 6,
            "blendshapes": 52,
        }
    model = FaceStreamClassifier(len(train_data.get_class_names()), stream_dims)
    trainer = SLRTrainer(model, train_loader, val_loader, device="cpu")
    results = trainer.train(epochs=args.epochs, verbose=True)
    print(
        "best validation accuracy={:.4f} at epoch {}".format(
            results["best_val_acc"], results["best_epoch"] + 1
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "stream_dims": stream_dims,
                "num_classes": model.num_classes,
                "face_space": args.face_space,
                "subset": args.subset,
                "seed": args.seed,
                "best_val_acc": results["best_val_acc"],
            },
            args.output,
        )
        print("saved checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
