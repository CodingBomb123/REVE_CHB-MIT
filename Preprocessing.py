import argparse
import os
import re

import mne
import numpy as np

mne.set_log_level("ERROR")

TARGET_SFREQ = 200

WINDOW_S = 10
PREICTAL_LEAD_S = 300


def parse_summary(summary_path):
    seizures_by_file = {}
    current_file = None
    with open(summary_path) as f:
        for line in f:
            line = line.strip()

            m = re.match(r"File Name:\s*(\S+)", line)
            if m:
                current_file = m.group(1)
                seizures_by_file[current_file] = []
                continue

            m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            if m:
                onset = int(m.group(1))
                seizures_by_file[current_file].append([onset, None])
                continue

            m = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            if m:
                offset = int(m.group(1))
                seizures_by_file[current_file][-1][1] = offset
                continue

    return {fn: [tuple(s) for s in sz] for fn, sz in seizures_by_file.items()}


def load_and_resample(edf_path, target_sfreq=TARGET_SFREQ):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)
    data = raw.get_data().astype(np.float32) * 1e6
    return data, raw.info["sfreq"], raw.ch_names


def slice_windows(data, sfreq, start_s, end_s, window_s=WINDOW_S):
    win_len = int(round(window_s * sfreq))
    start_idx = int(round(start_s * sfreq))
    end_idx = int(round(end_s * sfreq))
    n_windows = max(0, (end_idx - start_idx) // win_len)
    return [data[:, start_idx + i * win_len : start_idx + (i + 1) * win_len] for i in range(n_windows)]


def extract_preictal(data, sfreq, seizures, lead_s=PREICTAL_LEAD_S, window_s=WINDOW_S):
    windows = []
    for onset, _offset in seizures:
        pre_start = onset - lead_s
        if pre_start < 0:
            print(f"    WARNING: seizure at {onset}s has under {lead_s}s of lead-in "
                  f"({onset}s available) -- using the truncated window instead of skipping it.")
            pre_start = 0
        windows.extend(slice_windows(data, sfreq, pre_start, onset, window_s))
    return windows


def _file_num(fn):
    return int(re.search(r"_(\d+)\.edf$", fn).group(1))


def build_dataset(data_dir, summary_name, out_path):
    summary_path = os.path.join(data_dir, summary_name)
    seizures_by_file = parse_summary(summary_path)

    all_files = sorted(seizures_by_file.keys(), key=_file_num)
    seizure_files = {fn for fn, sz in seizures_by_file.items() if len(sz) > 0}

    excluded = set()
    for i, fn in enumerate(all_files):
        if fn in seizure_files:
            continue
        neighbors = [all_files[j] for j in (i - 1, i + 1) if 0 <= j < len(all_files)]
        if any(n in seizure_files for n in neighbors):
            excluded.add(fn)

    all_data, all_labels, all_sources = [], [], []
    ch_names_ref = None
    n_missing = 0

    for fn in all_files:
        edf_path = os.path.join(data_dir, fn)
        if not os.path.exists(edf_path):
            n_missing += 1
            continue

        print(f"Processing {fn} ...")
        data, sfreq, ch_names = load_and_resample(edf_path)

        if ch_names_ref is None:
            ch_names_ref = ch_names
        elif ch_names != ch_names_ref:
            raise ValueError(
                f"{fn} has a different channel set/order than earlier files.\n"
                f"  expected: {ch_names_ref}\n  got:      {ch_names}\n"
                "REVE needs a consistent channel order across every window."
            )

        seizures = seizures_by_file[fn]
        if seizures:
            windows = extract_preictal(data, sfreq, seizures)
            label = 1
            tag = "preictal"
        elif fn not in excluded:
            duration_s = data.shape[1] / sfreq
            windows = slice_windows(data, sfreq, 0, duration_s)
            label = 0
            tag = "interictal"
        else:
            print("  skipped (0-seizure file adjacent to a seizure file)")
            continue

        all_data.extend(windows)
        all_labels.extend([label] * len(windows))
        all_sources.extend([fn] * len(windows))
        print(f"  {len(windows)} {tag} windows")

    if n_missing:
        print(f"\n({n_missing} files listed in the summary aren't in {data_dir} yet -- skipped)")

    if not all_data:
        raise RuntimeError(f"No usable .edf files found in {data_dir}")

    X = np.stack(all_data)
    y = np.array(all_labels, dtype=np.int64)
    sources = np.array(all_sources)

    print(f"\nTotal: {len(y)} windows -- {int(y.sum())} preictal / {int((y == 0).sum())} interictal "
          f"({y.mean() * 100:.1f}% positive)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, X=X, y=y, sources=sources,
             ch_names=np.array(ch_names_ref), sfreq=TARGET_SFREQ)
    print(f"Saved to {out_path}  (X: {X.shape}, dtype {X.dtype})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with chb01_*.edf + chb01-summary.txt")
    ap.add_argument("--out", default="data/chb01_windows.npz")
    args = ap.parse_args()
    build_dataset(args.data_dir, "chb01-summary.txt", args.out)
