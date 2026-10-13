import argparse
import csv
import json
import math
import sys
import time
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
    f1_score,
)

# How to use: 
# Run the program in a terminal, typing "python {this code file path} {data folder file path}"
# Organise the data to be in one folder, with 3 subfolders named "train", "validation", "test"
# Set deep learning layer in class ConvBlock (ln62) and class BasicCSICNN (ln81)
# Set variables such as epoch, batch size, learning rate in the main function (ln227) by changing the default value
# Output folder is in the folder "Model", named as e.g.CNN_c12_w4000_s2000_b16_e32. c=class, w=window_width, s=window_stride, b=batch_size, e=total_epochs
# Outputs include: model of the best epoch, confusion matrix, epoch log, summary. Confusion matrix can be printed by using read_confusion_matrix.py 



class CSIdataset(Dataset):
    def __init__(
            self, 
            data, 
            ids, 
            packet_window: int = 4000, 
            packet_stride: int = 2000,
            subcarriers: int = 234
        ):
        self.data = data
        self.ids = ids
        self.packetN = packet_window
        self.packet_stride = packet_stride
        self.subcarriers = subcarriers

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        amp = self.data[idx]
        id = self.ids[idx]
        if amp.ndim ==2:
            amp = amp.unsqueeze(0)
        return {
            "X": amp, 
            "id": id
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,7), padding=(1,3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(3,5), padding=(1,2), bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=(1,4), stride=(1,2)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BasicCSICNN(nn.Module):
    """A simple CNN classifier for CSI amplitude maps.

    Input shape: [B, 234, 4000]
    """

    def __init__(self, in_channels, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16, pool=True),
            ConvBlock(16, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_loader(path_root: Path, data_type: Literal["train", "validation", "test"], label_to_id, batch_size):
    datatype_root = path_root / data_type
    npz_files = list(datatype_root.rglob("*.npz"))
    data, labels = [], []

    for file_path in npz_files:
        print(f"Loading: {file_path}")
        npz = np.load(file_path)
        print(npz["X"].shape)
        data.append(npz["X"])
        labels.append(npz["label_str"])

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(set(labels))
    ids = np.array([label_to_id[v] for v in labels], dtype = np.int64) 

    data = torch.from_numpy(data)
    ids = torch.from_numpy(ids)


    all_data = CSIdataset(data,ids)
    return DataLoader(dataset=all_data, batch_size=batch_size, shuffle = (data_type != "test"))


def metric_pack(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1s": f1_score(labels, preds, average=None, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, preds)
    }


def train_one_epoch(model, loader, device, optimizer, criterion, use_amp = True):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))
    loss_sum = 0.0
    labels_all, preds_all = [], []

    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        y = batch["id"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=(use_amp and device == "cuda")):
            scores = model(x)
            loss = criterion(scores,y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        labels_all.extend(y.detach().cpu().tolist())
        preds_all.extend(scores.argmax(dim=1).detach().cpu().tolist())

    metrics = metric_pack(labels_all, preds_all)
    metrics["loss"] = loss_sum / max(1, len(loader.dataset))
    return metrics


def evaluate_one_epoch(model, loader, device, criterion, num_classes, use_amp = True, with_paths=False):
    model.eval()
    loss_sum = 0.0
    labels_all, preds_all, paths_all = [], [], []

    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        y = batch["id"].to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda",enabled=(use_amp and device == "cuda")):
            scores = model(x)
            loss = criterion(scores,y)
        
        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        labels_all.extend(y.detach().cpu().numpy().tolist())
        preds_all.extend(scores.argmax(dim=1).detach().cpu().numpy().tolist())
        if with_paths:
            paths_all.extend(batch["path"])

    metrics = metric_pack(labels_all, preds_all)
    metrics["loss"] = loss_sum / max(1, len(loader.dataset))
    metrics["confusion_matrix"] = confusion_matrix(
        labels_all, preds_all, labels=list(range(num_classes))
    )
    metrics["labels"] = labels_all
    metrics["preds"] = preds_all
    if with_paths:
        metrics["paths"] = paths_all
    return metrics


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)

def append_csv(path: Path, row: dict, write_header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_root_folder", type = Path, help="One folder containing npz files or folders that contain npz files")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--classes", default=[
        "empty",
        "sitting",
        "standing",
        "normal_walking",
        "slow_walking",
        "sitting_down",
        "standing_up",
        "bending",
        "getting_up",
        "lying_down",
        "reaching",
        "falling"
    ])
    parser.add_argument("--packets", default = 4000)
    parser.add_argument("--packet_stride", default = 2000)
    parser.add_argument("--subcarriers", default = 234)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--lr", default=3e-4) #original learning rate
    parser.add_argument("--min_lr", default=1e-6) #minimum learning rate after scheduler steps
    parser.add_argument("--weight_decay", default=1e-4)
    parser.add_argument("--epochs", default=32)
    # parser.add_argument("--test_split", default=0.2) #fraction of total data being testing
    # parser.add_argument("--val_split", default=0.2) #fraction of total data being validation
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or (args.npz_root_folder.parent / "Model" / ("CNN_c" + str(len(args.classes))+ "_w" + str(args.packets) + "_s" + str(args.packet_stride) + "_b" + str(args.batch_size) + "_e" + str(args.epochs)))
    output_dir.mkdir(parents=True, exist_ok=True)


    # THIS DELETED CODE IS FOR UNSORTED DATA TO SPLIT INTO TRAIN, VAL, AND TEST 

    # npz_files = list(args.npz_root_folder.rglob("*.npz"))
    # data, labels = [], []
    # for file_path in npz_files:
    #     print(f"Loading: {file_path}")
    #     npz = np.load(file_path)
    #     print(npz["X"].shape)
    #     data.append(npz["X"])
    #     labels.append(npz["label_str"])

    # #concat all windows into one array of shape (N_data, packets, subcarriers)
    # data = np.concatenate(data, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # print(set(labels))

    # #change from string labels to int ids
    # label_to_id = {name: i for i, name in enumerate(args.classes)}
    # ids = np.array([label_to_id[v] for v in labels], dtype = np.int64) 

    # data = torch.from_numpy(data)
    # ids = torch.from_numpy(ids)


    # full_dataset = CSIdataset(data,ids)
    # N_data = len(full_dataset)

    # #split into test and all_train
    # N_test = int(N_data*args.test_split)
    # N_all_train = N_data - N_test
    # all_train_data, test_data = random_split(full_dataset, [N_all_train, N_test])

    # #split into val and train
    # N_val = int(N_data*args.val_split)
    # N_train = N_all_train - N_val
    # train_data, val_data = random_split(all_train_data, [N_train, N_val])

    # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    label_to_id = {name: i for i, name in enumerate(args.classes)}

    val_loader = get_loader(args.npz_root_folder, "validation", label_to_id, args.batch_size)
    test_loader = get_loader(args.npz_root_folder, "test", label_to_id, args.batch_size)
    train_loader = get_loader(args.npz_root_folder, "train", label_to_id, args.batch_size)


    model = BasicCSICNN(in_channels=1, num_classes=len(args.classes))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )


    #================TRAINING===================

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = output_dir / "best_basic_cnn_by_val_macro_f1.pth"
    log_path = output_dir / "epoch_log.csv"
    start_time = time.perf_counter()
    print("Starting Training:")

    for epoch in range(1,args.epochs+1):
        train_metrics = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = evaluate_one_epoch(model, val_loader, device, criterion, len(args.classes))
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"LR={lr:.3e} | "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Acc={train_metrics['accuracy']:.4f} "
            f"F1={train_metrics['macro_f1']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} "
            f"Acc={val_metrics['accuracy']:.4f} "
            f"AvgF1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": best_val_f1,
                    "label_to_id": label_to_id
                },
                best_path,
            )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1s": val_metrics["micro_f1s"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        append_csv(log_path, row, write_header=(epoch == 1))


    checkpoint = torch.load(best_path, map_location=device)
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
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "training_time_sec": time.perf_counter() - start_time,
            "confusion_matrix": str(output_dir / "confusion_matrix.npy"),
            #"predictions": str(output_dir / "test_predictions.csv"),
        },
    )

    print("\n========== Final Test Evaluation ==========")
    print(f"Best epoch by val Macro-F1 : {best_epoch}")
    print(f"Best val Macro-F1          : {best_val_f1:.4f}")
    print(f"Test Accuracy              : {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1              : {test_metrics['macro_f1']:.4f}")
    print(f"Test Balanced Accuracy     : {test_metrics['balanced_accuracy']:.4f}")
    print(f"Outputs saved to           : {output_dir}")

if __name__ == "__main__":
    main()

