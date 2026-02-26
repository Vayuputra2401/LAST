"""
Preprocessed Data Health Check
=================================
Checks the output of preprocess_data.py and produces a GIF visualisation.

Checks performed
-----------------
1. File existence & size — all 9 expected files per split present
2. Shape integrity      — (N, 3, T=64, V=25, M=2) for every stream
3. Cross-stream shape   — joint / velocity / bone / bone_velocity all have same N
4. Label integrity      — length matches N, all values in [0, num_classes)
5. Class distribution   — all 60 classes present; imbalance < 5×
6. Value statistics     — no NaN / Inf; joint range reasonable (~[-3, 3])
7. Cross-stream math    — velocity ≈ diff(joint); bone ≈ child - parent
8. Non-zero frames      — at least 90 % of frames are non-zero per sample

Output
-------
  • Console: PASS / FAIL table for every check
  • tests/data_check_report.txt — full text report
  • tests/data_check.gif        — skeleton animation, 1 train + 1 val sample

Usage
------
  python tests/check_preprocessed_data.py
  python tests/check_preprocessed_data.py --split_type xview
  python tests/check_preprocessed_data.py --data_dir E:/LAST-60/data/processed
"""

import sys, os, argparse, pickle, traceback, io
from pathlib import Path
from datetime import datetime
import numpy as np

# Force UTF-8 output on Windows (cmd/PowerShell default to cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath('.'))

# ── NTU-25 skeleton connectivity (child, parent) — 0-indexed ──────────────
NTU_PAIRS = [
    (1, 0), (20, 1), (2, 20), (3, 2),           # Spine
    (4, 20), (5, 4), (6, 5), (7, 6),             # Left arm (shoulder→wrist)
    (21, 7), (22, 6),                              # Left hand tips + thumb
    (8, 20), (9, 8), (10, 9), (11, 10),           # Right arm
    (23, 11), (24, 10),                            # Right hand tips + thumb
    (12, 0), (13, 12), (14, 13), (15, 14),        # Left leg
    (16, 0), (17, 16), (18, 17), (19, 18),        # Right leg
]

SPINE_CHAIN   = {0, 1, 2, 3, 20}
LEFT_BODY     = {4, 5, 6, 7, 21, 22, 12, 13, 14, 15}
RIGHT_BODY    = {8, 9, 10, 11, 23, 24, 16, 17, 18, 19}

BONE_COLORS = {
    'spine': '#cccccc', 'left': '#4488ff', 'right': '#ff4444',
}

def _bone_color(c, p):
    if c in SPINE_CHAIN and p in SPINE_CHAIN:
        return BONE_COLORS['spine']
    if c in LEFT_BODY or p in LEFT_BODY:
        return BONE_COLORS['left']
    return BONE_COLORS['right']


# ── Action name lookup ──────────────────────────────────────────────────────
try:
    from src.data.ntu120_actions import NTU120_ACTIONS
    def action_name(label: int) -> str:
        return NTU120_ACTIONS.get(label, f"class_{label}")
except ImportError:
    def action_name(label: int) -> str:
        return f"class_{label}"


# ── Reporting helpers ───────────────────────────────────────────────────────
_results = []

def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    line   = f"  [{status}] {name}"
    if detail:
        line += f"  =>  {detail}"
    _results.append((status, name, detail))
    print(line)
    return passed

def section(title: str):
    bar = "-" * 60
    msg = f"\n{bar}\n  {title}\n{bar}"
    print(msg)
    _results.append(("SECTION", title, ""))


# ── File discovery ──────────────────────────────────────────────────────────
STREAM_NAMES  = ["joint", "velocity", "bone", "bone_velocity"]
SPLITS        = ["train", "val"]

def expected_files(data_dir: Path, split_type: str):
    d = data_dir / split_type
    files = {}
    for split in SPLITS:
        files[f"{split}_label"] = d / f"{split}_label.pkl"
        for s in STREAM_NAMES:
            files[f"{split}_{s}"] = d / f"{split}_{s}.npy"
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Main checks
# ─────────────────────────────────────────────────────────────────────────────

def run_checks(data_dir: Path, split_type: str, num_classes: int = 60):
    all_passed = True

    # ── 1. File existence & size ─────────────────────────────────────────────
    section("1. File Existence & Size")
    files = expected_files(data_dir, split_type)
    for key, fpath in files.items():
        exists = fpath.exists()
        if exists:
            mb = fpath.stat().st_size / 1024**2
            ok = check(f"exists: {fpath.name}", True, f"{mb:.1f} MB")
        else:
            ok = check(f"exists: {fpath.name}", False, "FILE MISSING")
        all_passed &= ok

    # Abort if any files missing — rest of checks are meaningless
    missing = [k for k, f in files.items() if not f.exists()]
    if missing:
        print(f"\n  *** Aborting: {len(missing)} files missing. Run preprocess_data.py first.")
        return False, None, None

    # ── 2. Shape integrity ────────────────────────────────────────────────────
    section("2. Array Shapes  (expected: N × 3 × 64 × 25 × 2)")
    shapes = {}
    d = data_dir / split_type
    for split in SPLITS:
        for s in STREAM_NAMES:
            arr = np.load(d / f"{split}_{s}.npy", mmap_mode='r')
            shapes[f"{split}_{s}"] = arr.shape
            N, C, T, V, M = arr.shape
            ok = (C == 3 and T == 64 and V == 25 and M == 2)
            check(
                f"{split}_{s}.shape",
                ok,
                f"{arr.shape}  {'OK' if ok else 'WRONG - expected (N,3,64,25,2)'}"
            )
            all_passed &= ok

    # ── 3. Cross-stream N alignment ──────────────────────────────────────────
    section("3. Cross-Stream Sample Count Alignment")
    for split in SPLITS:
        ns = [shapes[f"{split}_{s}"][0] for s in STREAM_NAMES]
        ok = len(set(ns)) == 1
        check(f"{split}: all streams same N", ok, str(dict(zip(STREAM_NAMES, ns))))
        all_passed &= ok

    # ── 4. Labels ─────────────────────────────────────────────────────────────
    section(f"4. Labels  (expected: list of ints in [0, {num_classes-1}])")
    label_data = {}
    for split in SPLITS:
        lpath = d / f"{split}_label.pkl"
        with open(lpath, 'rb') as f:
            raw = pickle.load(f)
        if isinstance(raw, tuple):
            labels = list(raw[1])
        else:
            labels = list(raw)
        label_data[split] = labels
        N_arr = shapes[f"{split}_joint"][0]

        ok_len   = len(labels) == N_arr
        ok_range = all(0 <= l < num_classes for l in labels)
        ok_neg1  = sum(1 for l in labels if l == -1) == 0

        check(f"{split}: N_labels == N_array",  ok_len,   f"{len(labels)} vs {N_arr}")
        check(f"{split}: all labels in range",  ok_range, f"min={min(labels)}  max={max(labels)}")
        check(f"{split}: no -1 sentinel labels", ok_neg1,  f"{sum(1 for l in labels if l==-1)} bad")
        all_passed &= ok_len and ok_range and ok_neg1

    # ── 5. Class distribution ─────────────────────────────────────────────────
    section(f"5. Class Distribution  (all {num_classes} classes present?)")
    for split in SPLITS:
        labels = label_data[split]
        counts = np.bincount(labels, minlength=num_classes)
        present = int((counts > 0).sum())
        min_c, max_c = int(counts.min()), int(counts.max())
        ratio = max_c / max(min_c, 1)
        ok_all   = present == num_classes
        ok_ratio = ratio < 5.0    # no class has >5× samples vs least-populated

        check(f"{split}: all {num_classes} classes present", ok_all,
              f"{present}/{num_classes} classes found")
        check(f"{split}: imbalance < 5×", ok_ratio,
              f"min={min_c}  max={max_c}  ratio={ratio:.1f}×")
        all_passed &= ok_all

        # Show 5 least-populated classes
        bottom5 = np.argsort(counts)[:5]
        print(f"    Least-populated: " +
              ", ".join(f"{action_name(c)} (n={counts[c]})" for c in bottom5))

    # ── 6. Value statistics (sampled) ─────────────────────────────────────────
    section("6. Value Statistics  (NaN / Inf / range check on 200 random samples)")
    rng = np.random.default_rng(42)
    for split in SPLITS:
        joint_mm = np.load(d / f"{split}_joint.npy", mmap_mode='r')
        N = joint_mm.shape[0]
        idx = rng.choice(N, min(200, N), replace=False)
        chunk = np.array(joint_mm[idx])          # (200, 3, 64, 25, 2)

        nan_count = int(np.isnan(chunk).sum())
        inf_count = int(np.isinf(chunk).sum())
        vmin, vmax = float(chunk.min()), float(chunk.max())
        mean, std  = float(chunk.mean()), float(chunk.std())

        ok_nan   = nan_count == 0
        ok_inf   = inf_count == 0
        ok_range = (-10 < vmin) and (vmax < 10)

        check(f"{split}: no NaN",              ok_nan,   f"{nan_count} NaN values")
        check(f"{split}: no Inf",              ok_inf,   f"{inf_count} Inf values")
        check(f"{split}: joint range [-10,10]",ok_range, f"min={vmin:.3f}  max={vmax:.3f}  "
                                                          f"mean={mean:.3f}  std={std:.3f}")
        all_passed &= ok_nan and ok_inf and ok_range

        # Non-zero frame fraction
        # A frame is "active" if at least one joint has non-zero value
        frame_norms = np.abs(chunk[:, :, :, :, 0]).sum(axis=(1, 3))  # (200, T)
        nonzero_frac = float((frame_norms > 0).mean())
        ok_nonzero = nonzero_frac > 0.85
        check(f"{split}: non-zero frame fraction", ok_nonzero,
              f"{nonzero_frac*100:.1f}%  (threshold: 85%)")
        all_passed &= ok_nonzero

    # ── 7. Cross-stream math verification (5 samples) ─────────────────────────
    section("7. Cross-Stream Math  (velocity ~= diff(joint);  bone ~= child - parent)")
    for split in SPLITS:
        joint_mm    = np.load(d / f"{split}_joint.npy",    mmap_mode='r')
        vel_mm      = np.load(d / f"{split}_velocity.npy", mmap_mode='r')
        bone_mm     = np.load(d / f"{split}_bone.npy",     mmap_mode='r')
        N           = joint_mm.shape[0]
        idx5        = rng.choice(N, 5, replace=False)

        # velocity[:,  :-1, :, :] ≈ joint[:, 1:, :, :] - joint[:, :-1, :, :]
        vel_errors  = []
        bone_errors = []
        for i in idx5:
            j = np.array(joint_mm[i])    # (3, 64, 25, 2)
            v = np.array(vel_mm[i])
            b = np.array(bone_mm[i])

            expected_vel = np.zeros_like(j)
            expected_vel[:, :-1] = j[:, 1:] - j[:, :-1]
            vel_err = float(np.abs(v - expected_vel).max())
            vel_errors.append(vel_err)

            expected_bone = np.zeros_like(j)
            for child, parent in NTU_PAIRS:
                expected_bone[:, :, child, :] = j[:, :, child, :] - j[:, :, parent, :]
            bone_err = float(np.abs(b - expected_bone).max())
            bone_errors.append(bone_err)

        vel_ok  = all(e < 1e-4 for e in vel_errors)
        bone_ok = all(e < 1e-4 for e in bone_errors)
        check(f"{split}: velocity == diff(joint)",
              vel_ok, f"max_abs_err={max(vel_errors):.6f}")
        check(f"{split}: bone == child-parent",
              bone_ok, f"max_abs_err={max(bone_errors):.6f}")
        all_passed &= vel_ok and bone_ok

    return all_passed, label_data, d


# ─────────────────────────────────────────────────────────────────────────────
# GIF visualisation
# ─────────────────────────────────────────────────────────────────────────────

def make_gif(data_dir_split: Path, label_data: dict, out_path: Path, fps: int = 12):
    """
    Render a side-by-side skeleton animation GIF.
    Top row:    one TRAIN sample  (picked near the dataset midpoint)
    Bottom row: one VAL sample    (picked near the dataset midpoint)
    Front view: X (left/right) vs Z (up/down).
    Side view:  Y (depth)      vs Z (up/down).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("\n  [SKIP] matplotlib not found — skipping GIF generation")
        return

    print("\n  Generating skeleton GIF ...")

    joint_files = {
        'train': data_dir_split / 'train_joint.npy',
        'val':   data_dir_split / 'val_joint.npy',
    }

    samples_data   = {}
    samples_labels = {}

    for split in ['train', 'val']:
        arr = np.load(joint_files[split], mmap_mode='r')
        N   = arr.shape[0]
        # Pick a sample near the midpoint; avoid the very first samples
        # (those tend to cluster in the same class after sorting)
        pivot = N // 2
        idx   = pivot
        # Walk forward until we find a sample that has reasonable motion (std > 0.01)
        for candidate in range(pivot, min(pivot + 200, N)):
            s = np.array(arr[candidate, :, :, :, 0])  # (3, T, V)
            if s.std() > 0.01:
                idx = candidate
                break
        samples_data[split]   = np.array(arr[idx])   # (3, 64, 25, 2)
        samples_labels[split] = label_data[split][idx]

    T = samples_data['train'].shape[1]   # 64

    # ── figure layout ────────────────────────────────────────────────────────
    # 2 rows (train, val) × 2 cols (front view, side view)
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8),
        facecolor='#1a1a2e',
        gridspec_kw={'hspace': 0.35, 'wspace': 0.15}
    )
    fig.suptitle("Preprocessed Skeleton Data Check", color='white',
                 fontsize=13, fontweight='bold', y=0.98)

    ax_labels = {'front': 'Front view  (X vs Z)', 'side': 'Side view  (Y vs Z)'}

    for row, split in enumerate(['train', 'val']):
        data   = samples_data[split]   # (3, 64, 25, 2)
        label  = samples_labels[split]
        name   = action_name(label)
        body   = data[:, :, :, 0]     # (3, 64, 25) — body 0

        # Compute axis limits across all frames (front view: X vs Z; side: Y vs Z)
        x_all, y_all, z_all = body[0], body[1], body[2]
        xmin, xmax = x_all.min() - 0.1, x_all.max() + 0.1
        ymin, ymax = y_all.min() - 0.1, y_all.max() + 0.1
        zmin, zmax = z_all.min() - 0.1, z_all.max() + 0.1

        for col, view in enumerate(['front', 'side']):
            ax = axes[row, col]
            ax.set_facecolor('#0d0d1a')
            ax.set_aspect('equal')
            if view == 'front':
                ax.set_xlim(xmin, xmax); ax.set_ylim(zmin, zmax)
                ax.set_xlabel('X', color='#888', fontsize=7)
                ax.set_ylabel('Z', color='#888', fontsize=7)
            else:
                ax.set_xlim(ymin, ymax); ax.set_ylim(zmin, zmax)
                ax.set_xlabel('Y (depth)', color='#888', fontsize=7)
                ax.set_ylabel('Z', color='#888', fontsize=7)
            ax.tick_params(colors='#555', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            title_col = '#4488ff' if split == 'train' else '#ff8844'
            ax.set_title(
                f"[{split.upper()}]  {ax_labels[view]}",
                color=title_col, fontsize=8, pad=3
            )
        # Action name across both columns (centered)
        fig.text(0.5, 0.53 - row * 0.53,
                 f"Label {label}: \"{name}\"",
                 ha='center', va='top', color='white', fontsize=9,
                 fontweight='bold')

    # ── animation objects (one per row×view) ─────────────────────────────────
    drawn_bones  = [[[], []], [[], []]]   # [row][col][bone_lines]
    drawn_joints = [[None, None], [None, None]]
    frame_texts  = [[None, None], [None, None]]

    def _init_frame(frame_idx):
        artists = []
        for row, split in enumerate(['train', 'val']):
            body = samples_data[split][:, frame_idx, :, 0]  # (3, V)
            for col, view in enumerate(['front', 'side']):
                ax = axes[row, col]
                # clear old lines
                for ln in drawn_bones[row][col]:
                    ln.remove()
                drawn_bones[row][col] = []

                hx = body[0] if view == 'front' else body[1]  # horizontal
                vz = body[2]                                     # vertical

                # draw bones
                for (c, p) in NTU_PAIRS:
                    clr = _bone_color(c, p)
                    ln, = ax.plot([hx[p], hx[c]], [vz[p], vz[c]],
                                  color=clr, lw=1.5, solid_capstyle='round')
                    drawn_bones[row][col].append(ln)
                    artists.append(ln)

                # draw joints
                if drawn_joints[row][col] is not None:
                    drawn_joints[row][col].remove()
                sc = ax.scatter(hx, vz, s=14, c='white', zorder=5,
                                edgecolors='none', linewidths=0)
                drawn_joints[row][col] = sc
                artists.append(sc)

                # frame counter
                if frame_texts[row][col] is not None:
                    frame_texts[row][col].remove()
                txt = ax.text(
                    0.02, 0.96, f"frame {frame_idx+1:02d}/{T}",
                    transform=ax.transAxes, color='#aaaaaa',
                    fontsize=7, va='top'
                )
                frame_texts[row][col] = txt
                artists.append(txt)

        return artists

    # initialise at frame 0
    _init_frame(0)

    def animate(frame_idx):
        return _init_frame(frame_idx)

    ani = animation.FuncAnimation(
        fig, animate,
        frames=T,
        interval=1000 // fps,
        blit=False
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"  GIF saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(out_txt: Path, all_passed: bool):
    lines = [
        f"Preprocessed Data Health Check",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Overall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}",
        "",
    ]
    pass_count = sum(1 for s, _, _ in _results if s == 'PASS')
    fail_count = sum(1 for s, _, _ in _results if s == 'FAIL')
    lines.append(f"PASS: {pass_count}   FAIL: {fail_count}")
    lines.append("")
    for status, name, detail in _results:
        if status == 'SECTION':
            lines.append(f"\n{'-'*60}")
            lines.append(f"  {name}")
            lines.append(f"{'-'*60}")
        else:
            line = f"  [{status}] {name}"
            if detail:
                line += f"  =>  {detail}"
            lines.append(line)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n  Report saved → {out_txt}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Check preprocessed skeleton data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to processed data root (auto-detected if omitted)')
    parser.add_argument('--split_type', type=str, default='xsub',
                        choices=['xsub', 'xview', 'xset'],
                        help='Split type to check (default: xsub)')
    parser.add_argument('--env', type=str, default=None,
                        help='Environment name for auto-detection (default: auto)')
    parser.add_argument('--fps', type=int, default=12,
                        help='GIF frames per second (default: 12)')
    parser.add_argument('--no_gif', action='store_true',
                        help='Skip GIF generation')
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPROCESSED DATA HEALTH CHECK")
    print("=" * 60)

    # ── Resolve data directory ─────────────────────────────────────────────
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        try:
            from src.utils.config import load_config
            cfg     = load_config(env=args.env, dataset='ntu60')
            db      = cfg['environment']['paths']['data_base']
            data_dir = Path(db) / 'LAST-60' / 'data' / 'processed'
            print(f"  Auto-detected data dir: {data_dir}")
        except Exception as e:
            print(f"  [WARN] Could not auto-detect path ({e}), trying E:/LAST-60/data/processed")
            data_dir = Path('E:/LAST-60/data/processed')

    split_dir = data_dir / args.split_type
    print(f"  Split type : {args.split_type}")
    print(f"  Split dir  : {split_dir}")
    print()

    # ── Run checks ────────────────────────────────────────────────────────────
    try:
        all_passed, label_data, split_dir_path = run_checks(
            data_dir, args.split_type, num_classes=60
        )
    except Exception:
        print("\n[ERROR] Unexpected error during checks:")
        traceback.print_exc()
        sys.exit(2)

    # ── Summary ───────────────────────────────────────────────────────────────
    pass_count = sum(1 for s, _, _ in _results if s == 'PASS')
    fail_count = sum(1 for s, _, _ in _results if s == 'FAIL')
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {pass_count} passed  /  {fail_count} failed")
    if all_passed:
        print("  Overall: ALL CHECKS PASSED")
    else:
        print("  Overall: SOME CHECKS FAILED  (see above)")
    print('='*60)

    # ── Write report ──────────────────────────────────────────────────────────
    here = Path(__file__).parent
    write_report(here / 'data_check_report.txt', all_passed)

    # ── GIF ──────────────────────────────────────────────────────────────────
    if not args.no_gif and label_data is not None:
        try:
            make_gif(
                data_dir_split=split_dir_path,
                label_data=label_data,
                out_path=here / 'data_check.gif',
                fps=args.fps,
            )
        except Exception:
            print("\n  [WARN] GIF generation failed:")
            traceback.print_exc()

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
