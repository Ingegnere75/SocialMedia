import os
import json
import math
import argparse
import random
import platform
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import models
from transformers import AutoTokenizer, logging as hf_logging
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.dataset import SocialPostDataset, collate_fn
from src.model import SocialClassifier

hf_logging.set_verbosity_error()


# --------------------------- Utils ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backbone():
    # Feature extractor per immagini (se presenti). L'FC viene rimosso.
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    return resnet


def compute_class_stats_from_indices(dataset: SocialPostDataset, indices: np.ndarray):
    # Se il dataset espone il dataframe interno
    if hasattr(dataset, "df"):
        labels = dataset.df.iloc[indices]["label"].astype(int).to_numpy()
    else:
        # fallback (più lento): accesso singolo
        labels = []
        for i in indices:
            _, _, _, y = dataset[i]
            labels.append(int(y))
        labels = np.asarray(labels, dtype=int)

    num_classes = int(labels.max() + 1)
    counts = np.bincount(labels, minlength=num_classes)
    return labels, num_classes, counts


def make_class_weights(counts: np.ndarray):
    # pesi = inverso della frequenza (normalizzati)
    w = counts.sum() / (counts + 1e-8)
    w = w / w.sum()
    return w


# --------------------------- Train / Eval ---------------------------

def train_one_epoch(model, loader, optimizer, device, criterion, scaler, max_grad_norm=1.0):
    model.train()
    losses, all_y, all_p = [], [], []

    pbar = tqdm(loader, desc="train", leave=False)
    for enc, imgs, nums, labels in pbar:
        enc = {k: v.to(device) for k, v in enc.items()}
        imgs, nums, labels = imgs.to(device), nums.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=True):
            logits = model(enc, imgs, nums)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_p.extend(preds)
        all_y.extend(labels.detach().cpu().numpy())

        pbar.set_postfix(loss=np.mean(losses), acc=accuracy_score(all_y, all_p))

    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p, average="macro")
    return float(np.mean(losses)), acc, f1


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, all_y, all_p = [], [], []

    for enc, imgs, nums, labels in tqdm(loader, desc="eval", leave=False):
        enc = {k: v.to(device) for k, v in enc.items()}
        imgs, nums, labels = imgs.to(device), nums.to(device), labels.to(device)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=True):
            logits = model(enc, imgs, nums)
            loss = criterion(logits, labels)

        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_p.extend(preds)
        all_y.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p, average="macro")
    return float(np.mean(losses)), acc, f1, (np.asarray(all_y), np.asarray(all_p))


# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--num_classes", type=int, required=True)

    # training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # text/image backbone
    parser.add_argument("--text_model", type=str, default="cardiffnlp/twitter-roberta-base-sentiment")
    parser.add_argument("--freeze_text", type=str, default="true")
    parser.add_argument("--freeze_image", type=str, default="true")

    # dataloader / system
    parser.add_argument("--num_workers", type=int, default=0)  # default 0 per Windows
    parser.add_argument("--balance", type=str, choices=["none", "weights", "sampler", "both"], default="both")

    # checkpoint & logging
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--patience", type=int, default=2)  # early stopping
    args = parser.parse_args()

    # Windows: forziamo num_workers=0 se necessario
    if platform.system().lower().startswith("win") and args.num_workers != 0:
        print("[!] Windows rilevato: imposto --num_workers 0 (multiprocessing dataloader può fallire su Win).")
        args.num_workers = 0

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    ds = SocialPostDataset(
        args.csv_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        numeric_cols=None,
        img_size=args.img_size,
    )

    # split
    n = len(ds)
    n_val = max(1, int(n * args.val_split))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # collate
    collate = lambda batch: collate_fn(batch, tokenizer, max_length=args.max_length)

    # ------- Bilanciamento -------
    sampler = None
    criterion_weight = None

    # statistiche classi sul TRAIN
    train_indices = np.array(train_ds.indices)
    train_labels, num_classes_auto, class_counts = compute_class_stats_from_indices(ds, train_indices)

    if args.num_classes != num_classes_auto:
        print(f"[WARN] --num_classes={args.num_classes} ma nel train ho rilevato {num_classes_auto} classi.")
    print(f"[INFO] Distribuzione train: {class_counts.tolist()}")

    if args.balance in ("weights", "both"):
        cw = make_class_weights(class_counts)
        criterion_weight = torch.tensor(cw, dtype=torch.float32).to(device)
        print(f"[INFO] Class weights (loss): {cw.tolist()}")

    if args.balance in ("sampler", "both"):
        weights_per_class = 1.0 / (class_counts + 1e-8)
        sample_weights = weights_per_class[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        print("[INFO] WeightedRandomSampler attivo")

    # dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    # modello
    backbone = build_backbone()
    model = SocialClassifier(
        text_model_name=args.text_model,
        num_numeric=len(ds.numeric_cols) if getattr(ds, "numeric_cols", None) is not None else 0,
        num_classes=args.num_classes,
        text_proj=768,
        img_proj=512,
        num_proj=64,
        freeze_text=(args.freeze_text.lower() == "true"),
        freeze_image=(args.freeze_image.lower() == "true"),
        backbone=backbone,
    ).to(device)

    # loss / optim / scheduler
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    # OneCycleLR per convergenza più stabile su pochi epoch
    steps_per_epoch = math.ceil(len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # training loop + early stopping
    best_f1, best_path = -1.0, run_dir / "best.pt"
    patience_left = args.patience

    history = {"epoch": [], "train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler, args.grad_clip)
        scheduler.step()

        va_loss, va_acc, va_f1, (y_true, y_pred) = evaluate(model, val_loader, device, criterion)

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        print(f"Epoch {ep}: train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "label_map": getattr(ds, "idx_to_class", None),
                },
                best_path,
            )
            print(f"[+] Saved new best to {best_path} with F1={best_f1:.3f}")
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[INFO] Early stopping")
                break

    # report finale su validation
    report = classification_report(y_true, y_pred, digits=3, output_dict=False)
    cm = confusion_matrix(y_true, y_pred)
    print("Validation report:\n", report)
    print("Confusion matrix:\n", cm)

    # salva artefatti
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savetxt(run_dir / "confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")

    # dump anche il report in txt
    (run_dir / "val_report.txt").write_text(
        f"{report}\n\nConfusion matrix:\n{cm}\nBest F1: {best_f1:.4f}\n"
    )

    print(f"[INFO] Artefatti salvati in: {run_dir.as_posix()}")


if __name__ == "__main__":
    main()
