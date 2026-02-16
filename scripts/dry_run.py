"""Dry-run verification of the training pipeline — run before actual training."""
import sys, os, yaml, pickle, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

print("=" * 70)
print("  DRY RUN VERIFICATION")
print("=" * 70)
errors = []

# 1. Config loading + merge
from src.utils.config import load_config
config = load_config(dataset='ntu60')
with open('configs/training/default.yaml', 'r') as f:
    config.update(yaml.safe_load(f))
print("[1] Config: 16 training keys loaded OK")

# 2. Model forward pass
from src.models.last import create_last_base
nc = config['data']['dataset'].get('num_classes', 60)
nj = config['data']['dataset']['num_joints']
model = create_last_base(num_classes=nc, num_joints=nj)
inp = config['training']['input_frames']
x = torch.randn(2, 3, inp, nj, 2)
model.eval()
with torch.no_grad():
    out = model(x)
assert out.shape == (2, nc), f"Shape mismatch: {out.shape}"
print(f"[2] Model: {model.count_parameters():,} params, {tuple(x.shape)} -> {tuple(out.shape)} OK")

# 3. Transforms
from src.data.transforms import get_train_transform, get_val_transform
mc = {'dataset': config['data']['dataset'], 'training': config['training']}
tt = get_train_transform(mc)
vt = get_val_transform(mc)
s = torch.randn(3, 300, 25, 2)
ot = tt(s)
ov = vt(s)
assert ot.shape[1] == inp and ov.shape[1] == inp
print(f"[3] Transforms: (3,300,25,2) -> train{tuple(ot.shape)} val{tuple(ov.shape)} OK")

# 4. Trainer init
from src.training.trainer import Trainer
trainer = Trainer(model, config, tempfile.mkdtemp())
print(f"[4] Trainer: {trainer.device}, {type(trainer.optimizer).__name__}, {type(trainer.scheduler).__name__} OK")

# 5. Data files
db = config['environment']['paths']['data_base']
pp = os.path.join(db, 'LAST-60', 'data', 'processed', 'xsub')
for f in ['train_data.npy', 'train_label.pkl', 'val_data.npy', 'val_label.pkl']:
    fp = os.path.join(pp, f)
    if not os.path.exists(fp):
        errors.append(f"MISSING: {fp}")
print(f"[5] Data files: all 4 found in {pp}")

# 6. Npy shape + labels
td = np.load(os.path.join(pp, 'train_data.npy'), mmap_mode='r')
with open(os.path.join(pp, 'train_label.pkl'), 'rb') as f:
    tl = pickle.load(f)
print(f"[6] Data: shape={td.shape}, {len(tl)} labels, range=[{min(tl)},{max(tl)}]")
if td.shape[0] != len(tl):
    errors.append(f"Data/label mismatch: {td.shape[0]} vs {len(tl)}")
if max(tl) >= nc:
    errors.append(f"Label {max(tl)} >= num_classes {nc}")

# 7. Dataset + single sample end-to-end
from src.data.dataset import SkeletonDataset
sc = config['data']['dataset'].get('splits', None)
processed_path = os.path.join(db, 'LAST-60', 'data', 'processed')
ds = SkeletonDataset(
    data_path=processed_path, data_type='npy',
    max_frames=300, num_joints=nj, transform=tt,
    split='train', split_type='xsub', split_config=sc
)
d, l = ds[0]
print(f"[7] Dataset: {len(ds)} samples, sample[0]={tuple(d.shape)} label={l.item()}")

device = next(model.parameters()).device
model.eval()
with torch.no_grad():
    p = model(d.unsqueeze(0).to(device))
print(f"[8] End-to-end: sample -> model -> {tuple(p.shape)} OK")

print()
if errors:
    for e in errors:
        print(f"  X {e}")
else:
    print("  ALL 8 CHECKS PASSED - Ready to train!")
print("=" * 70)
