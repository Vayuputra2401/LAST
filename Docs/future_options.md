# V10 — Future Architectural Options & Competitive Analysis

## Additional Techniques for Small/Large

### 1. Per-Channel Topology (CTR-GCN) — High Impact, Medium Cost
CTR-GCN's key innovation: instead of K=3 shared adjacency matrices, it learns a **separate graph per channel**. Each of the 128/256/384 channels gets its own attention over joints.

```
Standard GCN:  A (K×V×V) shared for all channels
CTR-GCN:       A_c (K×V×V) per channel c → effectively K×C×V×V corrections
```

**Impact**: ~+1.5–2pp. This is what takes CTR-GCN to 92.4%.
**Cost**: Adds a channel-topology correction module (C×V×V learnable mask per block) — significant params but worth it for large.
**Verdict**: Add to **large only**. Too expensive for nano/small.

---

### 2. Dynamic Graph from Feature Similarity — Medium Impact, Low Cost
Compute adjacency dynamically per sample based on cosine similarity of joint features. Already exists in our LAST-E codebase as `A_dynamic`. Each sample gets a slightly different graph based on what it's actually doing.

```
A_dynamic = softmax(Q @ K^T / sqrt(d)) where Q,K = f(joint_features)
```

**Impact**: ~+0.5–1pp.
**Cost**: Small — embed_dim=C//4, gated by alpha_dyn init=-4.0 (starts inactive).
**Verdict**: Add to **small and large**. We already have the implementation in `light_gcn.py`.

---

### 3. Cross-Stream Feature Interaction — Medium Impact, Novel
Currently the 4 streams are completely independent through the backbone — they only interact at the classifier ensemble. Adding lightweight cross-stream attention at the end of each stage lets streams share information:

```
After stage output: joint features attend to bone features
                    bone features attend to velocity features
→ each stream's representation is enriched by the others
```

**Impact**: Estimated +0.5–1pp — streams are genuinely complementary (velocity captures timing, bone captures geometry).
**Cost**: 4×4 cross-stream attention weight per stage = very cheap.
**Verdict**: Add to **small and large**. Novel — no model in this class does this.

---

### 4. Hierarchical Joint Grouping (HDGCN-style) — Medium Impact
HDGCN computes representations at two scales: individual joints AND anatomical limb groups (treat the whole arm as one node). The limb-level and joint-level representations are then fused.

**Impact**: ~+0.5pp (HDGCN reports 93.4% at ~2M params vs 93.0% InfoGCN).
**Cost**: Requires a separate pooling operation per body part and an unpooling/broadcast back.
**Verdict**: Worth exploring for **large only**.

---

### 5. Spatial Self-Attention over Joints — Low-Medium Impact
InfoGCN uses an attention map over joints to weight which joints are most informative per action. Essentially a soft joint selector.

```
attn_joints = softmax(Linear(feat).T @ Linear(feat)) → (V×V)
```

**Impact**: ~+0.3–0.5pp.
**Cost**: 2×C×V params per block, very lightweight.
**Verdict**: Add to **small and large**.

---

## Where V10 Stands vs Competitors

### Nano tier (~200–300K params)

| Model | Params | NTU-60 xsub | Status |
|-------|--------|-------------|--------|
| **EfficientGCN-B0** | 290K | **90.2%** | Only published model in this tier |
| **V10 nano (target)** | 236K | **87–89% (no KD) / 88–90% (with KD)** | Our model, 20% fewer params |

**This is the prize fight.** EfficientGCN-B0 is the only published model at this param scale. If V10 nano hits 89%+ with 236K params, it beats B0 at 54K fewer parameters — a genuine efficiency claim.

---

### Small tier (~500K–1M params)

| Model | Params | NTU-60 xsub |
|-------|--------|-------------|
| Shift-GCN | ~750K | 90.7% |
| EfficientGCN-B4 | 1.1M | 92.1% |
| **V10 small** | **1.48M** | **targeting 90–92%** |

V10 small is currently at 1.48M — it has crept above the intended tier due to per-block GCN and StreamBN. Honest position: it competes with EfficientGCN-B4 (1.1M), not the "small" tier. **Shift-GCN at 750K, 90.7% is the more natural target to beat.** If V10 small hits 92%+ at 1.48M, it is comparable to EfficientGCN-B4 and edges toward CTR-GCN territory.

---

### Large tier (~1.5–3M params)

| Model | Params | NTU-60 xsub |
|-------|--------|-------------|
| CTR-GCN | 1.47M | 92.4% |
| InfoGCN | 1.63M | 93.0% |
| MS-G3D | 3.2M | 91.5% |
| **V10 large** | **3.18M** | **targeting 92–93%** |

V10 large at 3.18M competes directly with MS-G3D (3.2M, 91.5%). If V10 large reaches 92.5%+, it beats MS-G3D at the same param count. Matching InfoGCN (93.0%) at twice the params would be a weak result — CTR-GCN does it at 1.47M. Adding per-channel topology (technique #1 above) to large specifically targets this gap.

---

## The Honest Competitive Claim

| Claim | Required Result | Realistic? |
|-------|----------------|-----------|
| "Beat EfficientGCN-B0 at fewer params" | nano > 90.2% with KD, 236K | Ambitious, needs KD |
| "Match EfficientGCN-B0 at fewer params" | nano ≥ 89%, 236K | Realistic |
| "Beat Shift-GCN" | small > 90.7%, any params | Very likely |
| "Match EfficientGCN-B4" | small ≥ 92.1% | Possible with KD |
| "Beat MS-G3D at same params" | large > 91.5% | Likely |
| "Match CTR-GCN" | large ≥ 92.4% | With per-channel topology |

**The strongest paper claim if results hold**: "V10 nano achieves 89%+ at 236K parameters — matching EfficientGCN-B0's accuracy tier at 20% fewer parameters, using three-level anatomical grounding as the key inductive bias." That is a clean, verifiable, publishable claim.

The per-channel topology and dynamic graph for large/small are the next architectural levers if the base V10 results fall short.
