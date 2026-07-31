import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def parse_bipolar_channel(name):
    m = re.match(r"^([A-Za-z0-9]+)-([A-Za-z0-9]+?)(?:-\d+)?$", name)
    if not m:
        raise ValueError(f"Can't parse bipolar channel name: {name!r}")
    return m.group(1), m.group(2)


def _all_position_names(pos_bank):
    all_positions = pos_bank.get_all_positions()
    return list(all_positions.keys()) if isinstance(all_positions, dict) else list(all_positions)


def build_channel_positions(ch_names, pos_bank):
    known_names = _all_position_names(pos_bank)
    lookup = {n.upper(): n for n in known_names}

    def resolve(raw):
        key = raw.upper()
        if key not in lookup:
            raise KeyError(
                f"Electrode {raw!r} (parsed from a CHB-MIT channel name) was not "
                f"found in REVE's position bank. First 15 known names: {known_names[:15]}. "
                "This usually means a naming-convention mismatch (e.g. old vs new "
                "10-20 nomenclature), print pos_bank.get_all_positions() and compare."
            )
        return lookup[key]

    pairs = [tuple(resolve(e) for e in parse_bipolar_channel(ch)) for ch in ch_names]
    unique_electrodes = sorted({e for pair in pairs for e in pair})

    coords = pos_bank(unique_electrodes)
    coord_of = dict(zip(unique_electrodes, coords))

    midpoints = torch.stack([(coord_of[a] + coord_of[b]) / 2 for a, b in pairs])
    return midpoints


def reve_features(model, eeg, pos):
    with torch.no_grad():
        out_layers = model(eeg, pos, return_output=True)
    flat = out_layers[-1]
    b, seq_len, e = flat.shape
    c = pos.shape[1]
    h = seq_len // c
    return flat.view(b, c, h, e)


class WindowDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_collate_fn(positions):

    def collate(batch):
        xb = torch.stack([b[0] for b in batch])
        yb = torch.stack([b[1] for b in batch])
        pb = positions.unsqueeze(0).repeat(len(batch), 1, 1)
        return xb, yb, pb

    return collate


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    all_targets, all_preds, all_probs = [], [], []
    for xb, yb, pb in loader:
        xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
        feats = reve_features(model, xb, pb)
        logits = model.final_layer(feats)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_targets.append(yb.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()[:, 1]

    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": auroc,
    }
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}, metrics
