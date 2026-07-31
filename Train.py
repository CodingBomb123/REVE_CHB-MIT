import argparse

import compat
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, set_seed

from common import RMSNorm, WindowDataset, build_channel_positions, evaluate_model, make_collate_fn, reve_features


def split_by_source(sources, y, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)

    def split_group(file_list, label):
        file_list = list(file_list)
        rng.shuffle(file_list)
        n = len(file_list)
        if n < 3:
            print(f"  WARNING: only {n} {label} source file(s) available -- can't "
                  f"carve out a real val/test split yet. Putting all of it in train; "
                  f"sync the rest of chb01 before trusting any val/test numbers.")
            return file_list, [], []
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        return file_list[n_test + n_val:], file_list[n_test:n_test + n_val], file_list[:n_test]

    pre_train, pre_val, pre_test = split_group(sorted(set(sources[y == 1])), "preictal")
    inter_train, inter_val, inter_test = split_group(sorted(set(sources[y == 0])), "interictal")

    train_files = set(pre_train) | set(inter_train)
    val_files = set(pre_val) | set(inter_val)
    test_files = set(pre_test) | set(inter_test)

    return (np.isin(sources, list(train_files)),
            np.isin(sources, list(val_files)),
            np.isin(sources, list(test_files)))


def build_classifier_head(model, example_x, example_pos, device):
    model.eval()
    out = reve_features(model, example_x.unsqueeze(0).to(device), example_pos.unsqueeze(0).to(device))
    flat_dim = int(np.prod(out.shape[1:]))
    print(f"REVE backbone output shape per window: {tuple(out.shape)} -> flattened dim {flat_dim}")

    return nn.Sequential(
        nn.Flatten(),
        RMSNorm(flat_dim),
        nn.Dropout(0.1),
        nn.Linear(flat_dim, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chb01_windows.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="chb01_head.pt")
    args = ap.parse_args()

    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    d = np.load(args.data, allow_pickle=True)
    X, y, sources, ch_names = d["X"], d["y"], d["sources"], list(d["ch_names"])

    class_counts = np.bincount(y, minlength=2)
    if (class_counts == 0).any():
        raise RuntimeError(
            f"Only one class present in {args.data} (interictal={class_counts[0]}, "
            f"preictal={class_counts[1]}). This happens when only one/a few chb01 "
            f"files have been synced so far -- re-run preprocessing.py once more of "
            f"chb01 is on disk, then retry."
        )

    train_mask, val_mask, test_mask = split_by_source(sources, y)
    print(f"windows -- train: {train_mask.sum()}  val: {val_mask.sum()}  test: {test_mask.sum()}")

    print("Loading REVE-base + position bank from HuggingFace "
          "(gated -- run `hf auth login` first if this fails with an auth error)...")
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, torch_dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, torch_dtype="auto")

    positions = build_channel_positions(ch_names, pos_bank)

    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    model.final_layer = build_classifier_head(model, torch.from_numpy(X[0]), positions, device).to(device)

    collate = make_collate_fn(positions)
    train_loader = DataLoader(WindowDataset(X[train_mask], y[train_mask]), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate)
    val_loader = (DataLoader(WindowDataset(X[val_mask], y[val_mask]), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)
                  if val_mask.sum() else None)

    train_counts = np.bincount(y[train_mask], minlength=2)
    class_weights = torch.tensor(len(y[train_mask]) / (2 * train_counts), dtype=torch.float32).to(device)
    print(f"train class counts: {train_counts}, loss weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.final_layer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_bacc, best_state = -1.0, None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb, pb in train_loader:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            optimizer.zero_grad()
            feats = reve_features(model, xb, pb)
            out = model.final_layer(feats)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            _, val_metrics = evaluate_model(model, val_loader, device)
            bacc = val_metrics["balanced_accuracy"]
            print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_balanced_acc={bacc:.4f}")
            if bacc > best_val_bacc:
                best_val_bacc = bacc
                best_state = {k: v.detach().clone() for k, v in model.final_layer.state_dict().items()}
            scheduler.step(bacc)
        else:
            print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  (no val set yet)")

    if best_state is not None:
        model.final_layer.load_state_dict(best_state)

    torch.save({
        "head": model.final_layer,
        "positions": positions,
        "ch_names": ch_names,
        "test_mask": test_mask,
    }, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
