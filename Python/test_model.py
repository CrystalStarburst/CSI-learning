import argparse
import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score
)

from model_training import (
    BasicCSICNN,
    get_loader,
    save_json,
    evaluate_one_epoch
)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type = Path, help="The path containing model state")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--classes", default=[
        "empty",
        "sitting",
        "standing",
        "normal_walking",
        "slow_walking",
        "sitting_down",
        "standing_up"
    ])
    # parser.add_argument("--packets", default = 4000)
    # parser.add_argument("--packet_stride", default = 2000)
    # parser.add_argument("--subcarriers", default = 234)
    # parser.add_argument("--batch_size", default=32)
    # parser.add_argument("--lr", default=1e-3) #original learning rate
    # parser.add_argument("--min_lr", default=1e-6) #minimum learning rate after scheduler steps
    # parser.add_argument("--weight_decay", default=1e-4)
    # parser.add_argument("--epochs", default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or (args.model_path.parent / "test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_root_folder = Path("C:/Users/user/csi/SplitData")

    label_to_id = {name: i for i, name in enumerate(args.classes)}
    test_loader = get_loader(npz_root_folder, "test", label_to_id, args.batch_size)

    
    model = BasicCSICNN(in_channels=1, num_classes=len(args.classes))
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_one_epoch(
        model,
        test_loader,
        device,
        criterion,
        len(args.classes)
    )

    np.save(output_dir / "confusion_matrix.npy", test_metrics["confusion_matrix"])

    save_json(
        output_dir / "summary.json",
        {
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "confusion_matrix": str(output_dir / "confusion_matrix.npy"),
            "predictions": str(output_dir / "test_predictions.csv"),
        },
    )

    print("\n========== Final Test Evaluation ==========")
    print(f"Test Accuracy              : {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1              : {test_metrics['macro_f1']:.4f}")
    print(f"Test Macro-F1              : {test_metrics['micro_f1s']:.4f}")
    print(f"Test Balanced Accuracy     : {test_metrics['balanced_accuracy']:.4f}")
    print(f"Outputs saved to           : {output_dir}")

if __name__ == "__main__":
    main()
    
    