"""Component-by-component diagnosis of the TKCE-joint pipeline.

The end-to-end number ("run the models in tandem") is disappointing but does not
say WHERE the signal is lost. This script opens the box and tests every stage of

    Tree  ->  Kernel K  ->  Siamese encoder phi  ->  head g  ->  prediction

in isolation, so the failure can be *localised* instead of guessed. It answers,
in order, the questions the professor named:

  Checkpoint 1  "is there any structure in the kernel?"
                -> kernel-target alignment, within/between-class similarity, and a
                   kNN classifier run DIRECTLY on the tree kernel K.
  Checkpoint 2  "after training the siamese, did it preserve?"
                -> compare the embedding Gram matrix Ghat = phi.phi^T against K:
                   alignment, entry correlation, and nearest-neighbour recall.
  Checkpoint 3  "is the embedding still useful for the label?"
                -> a linear probe and a kNN on the frozen embedding phi(X).
  Checkpoint 4  "does the head actually exploit it?"
                -> the full trained head vs the probes above.
  Collapse      contrastive encoders often map everything to one point; measure
                effective rank, dead dimensions, mean pairwise cosine.

Everything is stacked into ONE accuracy ladder so the biggest drop between two
rungs is the culprit:

    Tree ceiling  >  kNN on K  >  kNN on phi  ~ linear probe on phi  >  full head

The joint encoder is trained here with the same recipe as run_deep_joint.py so
the diagnosed phi is the one that produced the disappointing number.

Example (run on the GPU server):
    python run_diagnosis.py --task 361070 --epochs 400 --lr 1e-6 --lam 0.015 \
        --loss infonce --device auto
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

from tkce.baselines import fit_tree_baseline, leaf_onehot_features
from tkce.data import load_task
from tkce.kernels import build_kernel
from tkce.losses import UncertaintyWeighting, apply_pair_loss, build_contrastive
from tkce.metrics import clf_metrics
from tkce.models import JointModel, SiameseEncoder, build_head, head_out_dim
from tkce.pairs import AnchorPositiveDataset, SampledPositiveIndex
from tkce.train import resolve_device


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _cycle(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def embed(enc, X, device, bs=8192):
    """phi(X); rows are L2-normalised so an inner product is a cosine."""
    enc.eval()
    out = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        out.append(enc(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def head_proba(model, X, device, bs=8192):
    model.eval()
    ps = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        logits, _ = model(xb)
        ps.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ps, axis=0)


def alignment(A, B):
    """Centred kernel-target alignment in [-1, 1]: <A,B>_F / (||A||.||B||)."""
    a, b = A.ravel(), B.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def label_gram(y):
    """M_ij = 1 if same class else 0 — the ideal 'label kernel'."""
    return (y[:, None] == y[None, :]).astype(np.float32)


def knn_proba_from_similarity(S, y_train, n_classes, k):
    """kNN class-probabilities from a (n_query, n_train) SIMILARITY matrix."""
    k = min(k, S.shape[1])
    top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]     # (n_query, k)
    lab = y_train[top]                                      # neighbour labels
    proba = np.zeros((S.shape[0], n_classes), dtype=np.float64)
    for c in range(n_classes):
        proba[:, c] = (lab == c).mean(axis=1)
    return proba


def recall_at_k(K, G, k):
    """Fraction of each point's top-k K-neighbours that are also top-k under G,
    averaged over points (self excluded). Measures neighbourhood preservation."""
    n = K.shape[0]
    Kd, Gd = K.copy(), G.copy()
    np.fill_diagonal(Kd, -np.inf)
    np.fill_diagonal(Gd, -np.inf)
    kk = min(k, n - 1)
    kn = np.argpartition(-Kd, kth=kk - 1, axis=1)[:, :kk]
    gn = np.argpartition(-Gd, kth=kk - 1, axis=1)[:, :kk]
    hits = [len(set(kn[i]) & set(gn[i])) / kk for i in range(n)]
    return float(np.mean(hits))


def effective_rank(Z):
    """Participation ratio of phi's covariance eigenvalues: (sum l)^2 / sum l^2.
    Close to embedding_dim = healthy; close to 1 = collapsed to a line."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(len(Z) - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# train the joint encoder+head (same recipe as run_deep_joint.run_one)
# --------------------------------------------------------------------------- #
def fit_joint(ds, args, device):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    enc = SiameseEncoder(ds.n_features,
                         hidden_dims=tuple([args.enc_width] * args.enc_depth),
                         embedding_dim=args.embedding_dim,
                         dropout=args.enc_dropout).to(device)
    head_cfg = dict(hidden_dims=(args.d, args.d), dropout=args.dropout,
                    d=args.d, d_hidden=args.d_hidden, n_blocks=args.n_blocks)
    head = build_head(args.head, args.embedding_dim, head_out_dim(ds), head_cfg).to(device)
    model = JointModel(enc, head).to(device)

    contrast, regime = build_contrastive(args.loss, args.temperature,
                                         dim=args.embedding_dim, margin=args.margin)
    if regime != "anchorpos":
        raise SystemExit(f"'{args.loss}' is two-stage only; pick a joint loss.")
    contrast = contrast.to(device)

    # Kernel source. GBT is a boosted (residual-fitting, shallow, correlated)
    # ensemble; RF is bagged (independent, deep, class-pure leaves) — leaf
    # co-occupancy is really Random-Forest *proximity*, so RF/Mondrian match the
    # co-occupancy->Laplace theory that GBT does not. Swap here to test it.
    if args.kernel == "rf":
        kcfg = {"n_estimators": args.k_n_estimators, "max_depth": args.rf_max_depth,
                "min_samples_leaf": args.rf_min_leaf, "random_state": args.seed}
    elif args.kernel == "mondrian":
        kcfg = {"n_estimators": args.k_n_estimators, "max_depth": args.mondrian_depth,
                "random_state": args.seed}
    else:  # gbt (boosting)
        kcfg = {"n_estimators": args.k_n_estimators,
                "max_depth": args.k_max_depth, "random_state": args.seed}
    print(f"  [kernel] source={args.kernel}  cfg={kcfg}", flush=True)
    kern = build_kernel(args.kernel, ds.X_train, ds.y_train, ds.task_type, kcfg)
    idx = SampledPositiveIndex(kern, kern.leaves(ds.X_train),
                               pos_threshold=args.pos_threshold, max_pos=50, seed=args.seed)
    ap_ds = AnchorPositiveDataset(ds.X_train, idx, seed=args.seed)
    cbs = args.contrastive_batch_size or args.batch_size
    ap_gen = None
    if len(ap_ds) > 0 and args.lam > 0:
        ap_gen = _cycle(DataLoader(ap_ds, batch_size=cbs, shuffle=True, drop_last=False))
    print(f"  [encoder] positive-pair coverage = {idx.coverage():.1%} "
          f"({len(ap_ds)} anchors with >=1 positive)", flush=True)

    uw = UncertaintyWeighting(2).to(device) if args.uncertainty_weighting else None
    params = list(model.parameters()) + list(contrast.parameters())
    if uw is not None:
        params += list(uw.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sup_loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train).to(device),
                      torch.from_numpy(ds.y_train).to(device)),
        batch_size=args.batch_size, shuffle=True)

    con_curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        csum, nb = 0.0, 0
        for xb, yb in sup_loader:
            opt.zero_grad()
            out, _ = model(xb)
            t_loss = F.cross_entropy(out, yb)
            loss = t_loss
            if ap_gen is not None:
                xi, xj = next(ap_gen)
                c_loss = apply_pair_loss(args.loss, contrast,
                                         enc(xi.to(device)), enc(xj.to(device)))
                loss = (uw([t_loss, c_loss]) if uw is not None
                        else t_loss + args.lam * c_loss)
                csum += c_loss.item(); nb += 1
            loss.backward(); opt.step()
        con_curve.append(csum / max(nb, 1))
        if epoch % max(1, args.epochs // 6) == 0:
            print(f"    epoch {epoch:4d}/{args.epochs}  contrastive_loss="
                  f"{con_curve[-1]:.4f}", flush=True)
    return model, enc, kern, con_curve


# --------------------------------------------------------------------------- #
# main diagnosis
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070)          # eye_movements
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    # ---- joint-training recipe (mirror run_deep_joint defaults) ----
    ap.add_argument("--loss", default="infonce")
    ap.add_argument("--lam", type=float, default=0.015)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--contrastive-batch-size", type=int, default=256)
    ap.add_argument("--head", default="tabresnet", choices=["tabresnet", "mlp"])
    ap.add_argument("--enc-width", type=int, default=512)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--enc-dropout", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--uncertainty-weighting", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pos-threshold", type=float, default=0.6)
    ap.add_argument("--kernel", default="gbt", choices=["gbt", "rf", "mondrian"],
                    help="kernel SOURCE: gbt=boosting (default); rf=Random Forest "
                         "(bagged, deep, class-pure leaves = RF proximity); "
                         "mondrian=unsupervised random partitions")
    ap.add_argument("--k-n-estimators", type=int, default=200)
    ap.add_argument("--k-max-depth", type=int, default=4, help="depth for the GBT kernel")
    ap.add_argument("--rf-max-depth", type=int, default=None,
                    help="depth for the RF kernel (default: unlimited/deep — the point of RF)")
    ap.add_argument("--rf-min-leaf", type=int, default=5,
                    help="min_samples_leaf for the RF kernel")
    ap.add_argument("--mondrian-depth", type=int, default=6,
                    help="depth for the Mondrian kernel")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # ---- diagnostic knobs ----
    ap.add_argument("--diag-sample", type=int, default=2000,
                    help="points for the K vs Ghat matrix analysis (O(n^2) memory)")
    ap.add_argument("--knn-train", type=int, default=4000,
                    help="train anchors for the kNN ladder rungs")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/diagnosis")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    if ds.task_type != "classification":
        raise SystemExit(f"{ds.name} is not classification; diagnosis assumes clf.")
    rng = np.random.default_rng(args.seed)
    C = ds.n_classes
    print(f"\n=== DIAGNOSIS: {ds.name} (task {args.task}) | "
          f"{ds.n_features} feats, {C} classes | kernel={args.kernel} | "
          f"device={device} ===\n", flush=True)

    # -------- ceiling: native tree(s) --------
    print("[ceiling] fitting tree baselines ...", flush=True)
    xgb_te, _, _ = fit_tree_baseline("xgboost", ds, {"seed": args.seed})
    lgb_te, _, _ = fit_tree_baseline("lightgbm", ds, {"seed": args.seed})
    tree_auc = max(xgb_te["auc"], lgb_te["auc"])
    print(f"  xgboost  AUC={xgb_te['auc']:.4f}  acc={xgb_te['accuracy']:.4f}")
    print(f"  lightgbm AUC={lgb_te['auc']:.4f}  acc={lgb_te['accuracy']:.4f}  "
          f"-> ceiling {tree_auc:.4f}\n", flush=True)

    # -------- train the joint encoder+head we are diagnosing --------
    print(f"[train] joint encoder+head  loss={args.loss} lam={args.lam} "
          f"epochs={args.epochs} lr={args.lr:g}", flush=True)
    model, enc, kern, con_curve = fit_joint(ds, args, device)
    full_proba = head_proba(model, ds.X_test, device)
    full = clf_metrics(ds.y_test, full_proba)
    print(f"  full TKCE head  AUC={full['auc']:.4f}  acc={full['accuracy']:.4f}\n",
          flush=True)

    # -------- embeddings for every split --------
    phi_tr = embed(enc, ds.X_train, device)
    phi_te = embed(enc, ds.X_test, device)

    # =====================================================================
    # Checkpoint 1 — is there structure IN THE KERNEL?
    # =====================================================================
    print("[checkpoint 1] kernel structure", flush=True)
    # subsample train for the O(n^2) matrix work
    ntr = len(ds.X_train)
    s_idx = rng.choice(ntr, min(args.diag_sample, ntr), replace=False)
    Ls = kern.leaves(ds.X_train[s_idx])
    Ks = kern.kernel_matrix(Ls)                            # (m, m) tree kernel
    ys = ds.y_train[s_idx]
    M = label_gram(ys)
    same = M.astype(bool); np.fill_diagonal(same, False)
    diff = ~M.astype(bool)
    k_align_y = alignment(Ks, M)
    within = float(Ks[same].mean())
    between = float(Ks[diff].mean())
    print(f"  alignment(K, labels) = {k_align_y:.4f}   "
          f"(within-class K={within:.4f} vs between-class K={between:.4f}, "
          f"gap={within - between:+.4f})", flush=True)

    # kNN classifier run DIRECTLY on the kernel (does K predict the label?)
    kn_idx = rng.choice(ntr, min(args.knn_train, ntr), replace=False)
    L_knn = kern.leaves(ds.X_train[kn_idx])
    L_te = kern.leaves(ds.X_test)
    Kte = kern.kernel_matrix(L_te, L_knn)                  # (n_test, n_knn)
    knnK_proba = knn_proba_from_similarity(Kte, ds.y_train[kn_idx], C, args.knn_k)
    knnK = clf_metrics(ds.y_test, knnK_proba)
    print(f"  kNN ON KERNEL K:  AUC={knnK['auc']:.4f}  acc={knnK['accuracy']:.4f}\n",
          flush=True)

    # =====================================================================
    # Checkpoint 2 — did the SIAMESE preserve the kernel?
    # =====================================================================
    print("[checkpoint 2] kernel preservation by phi", flush=True)
    phi_s = phi_tr[s_idx]
    Gs = phi_s @ phi_s.T                                   # cosine Gram (normalised)
    tri = np.triu_indices_from(Ks, k=1)                    # off-diagonal pairs
    corr = float(np.corrcoef(Ks[tri], Gs[tri])[0, 1])
    g_align_K = alignment(Gs, Ks)
    g_align_y = alignment(Gs, M)
    recalls = {k: recall_at_k(Ks, Gs, k) for k in (5, 10, 20, 50)}
    print(f"  alignment(Ghat, K)     = {g_align_K:.4f}   (1.0 = perfect copy)")
    print(f"  corr(Ghat_ij, K_ij)    = {corr:.4f}")
    print(f"  alignment(Ghat, labels)= {g_align_y:.4f}   (vs K's {k_align_y:.4f})")
    print("  neighbourhood recall@k = " +
          ", ".join(f"@{k}:{v:.2f}" for k, v in recalls.items()) + "\n", flush=True)

    # =====================================================================
    # Checkpoint 3 — is the embedding USEFUL for the label?
    # =====================================================================
    print("[checkpoint 3] embedding usefulness (frozen phi)", flush=True)

    def probe(name, Ztr, Zte):
        lr = LogisticRegression(max_iter=2000, n_jobs=-1)
        lr.fit(Ztr, ds.y_train)
        p = lr.predict_proba(Zte)
        m = clf_metrics(ds.y_test, p)
        print(f"  linear probe on {name:<12} AUC={m['auc']:.4f}  "
              f"acc={m['accuracy']:.4f}")
        return m

    probe_phi = probe("phi", phi_tr, phi_te)
    probe_raw = probe("raw X", ds.X_train, ds.X_test)
    Ltr, _, Lte = leaf_onehot_features(ds, {"seed": args.seed})
    probe_leaf = probe("leaf-onehot", Ltr, Lte)

    # kNN on phi (same k as kNN-on-K -> apples-to-apples preservation-in-prediction)
    knn = KNeighborsClassifier(n_neighbors=args.knn_k, metric="cosine")
    knn.fit(phi_tr[kn_idx], ds.y_train[kn_idx])
    knnP_proba = knn.predict_proba(phi_te)
    # align proba columns to 0..C-1 (KNeighbors uses sorted classes -> already 0..C-1)
    knnPhi = clf_metrics(ds.y_test, knnP_proba)
    print(f"  kNN ON EMBEDDING phi:  AUC={knnPhi['auc']:.4f}  "
          f"acc={knnPhi['accuracy']:.4f}\n", flush=True)

    # =====================================================================
    # Collapse check
    # =====================================================================
    erank = effective_rank(phi_tr[s_idx])
    dim_std = phi_tr.std(axis=0)
    dead = int((dim_std < 1e-3).sum())
    off = Gs[tri]
    mean_abs_cos = float(np.abs(off).mean())
    print("[collapse] embedding health", flush=True)
    print(f"  effective rank       = {erank:.1f} / {args.embedding_dim}")
    print(f"  dead dims (std<1e-3) = {dead} / {args.embedding_dim}")
    print(f"  mean |pairwise cos|  = {mean_abs_cos:.3f}  (near 1.0 = collapsed)\n",
          flush=True)

    # =====================================================================
    # The ladder + verdict
    # =====================================================================
    ladder = [
        ("Tree (ceiling)", tree_auc),
        ("kNN on K", knnK["auc"]),
        ("kNN on phi", knnPhi["auc"]),
        ("linear probe on phi", probe_phi["auc"]),
        ("full TKCE head", full["auc"]),
        ("ref: leaf-onehot probe", probe_leaf["auc"]),
        ("ref: raw-X probe", probe_raw["auc"]),
    ]
    print("=" * 60)
    print(f"ACCURACY LADDER (test AUC) — {ds.name}")
    print("=" * 60)
    for nm, v in ladder:
        print(f"  {nm:<26} {v:.4f}")
    # localise the biggest drop among the core rungs
    core = ladder[:5]
    drops = [(core[i][0] + " -> " + core[i + 1][0], core[i][1] - core[i + 1][1])
             for i in range(len(core) - 1)]
    worst = max(drops, key=lambda t: t[1])
    print(f"\n  biggest drop: {worst[0]}  (-{worst[1]:.4f})")
    print("  ^ that transition is where the signal is being lost.\n", flush=True)

    # -------- persist numbers --------
    summary = dict(
        dataset=ds.name, task=args.task, seed=args.seed, device=str(device),
        kernel=args.kernel,
        tree_ceiling_auc=tree_auc, xgb_auc=xgb_te["auc"], lgb_auc=lgb_te["auc"],
        full_head_auc=full["auc"], full_head_acc=full["accuracy"],
        knn_on_K_auc=knnK["auc"], knn_on_phi_auc=knnPhi["auc"],
        probe_phi_auc=probe_phi["auc"], probe_raw_auc=probe_raw["auc"],
        probe_leaf_auc=probe_leaf["auc"],
        kernel_align_labels=k_align_y, within_class_K=within, between_class_K=between,
        Ghat_align_K=g_align_K, Ghat_K_corr=corr, Ghat_align_labels=g_align_y,
        recall_at_k=recalls, effective_rank=erank, embedding_dim=args.embedding_dim,
        dead_dims=dead, mean_abs_cosine=mean_abs_cos,
        biggest_drop=worst[0], biggest_drop_amount=worst[1],
        contrastive_loss_first=con_curve[0], contrastive_loss_last=con_curve[-1],
    )
    with open(os.path.join(args.out, f"diag_{ds.name}_{args.kernel}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(ladder, columns=["rung", "test_auc"]).to_csv(
        os.path.join(args.out, f"diag_{ds.name}_{args.kernel}_ladder.csv"), index=False)

    # -------- figure --------
    order = np.argsort(ys)                                 # sort matrices by label
    fig, ax = plt.subplots(3, 3, figsize=(18, 15))

    # (0,0) the ladder
    names = [n for n, _ in ladder]; vals = [v for _, v in ladder]
    cols = ["#2ca02c", "#1f77b4", "#1f77b4", "#1f77b4", "#d62728",
            "#7f7f7f", "#7f7f7f"]
    ax[0, 0].barh(range(len(names))[::-1], vals, color=cols, edgecolor="white")
    ax[0, 0].set_yticks(range(len(names))[::-1]); ax[0, 0].set_yticklabels(names, fontsize=9)
    ax[0, 0].set_xlim(0.5, max(0.75, tree_auc + 0.02))
    ax[0, 0].axvline(tree_auc, ls="--", c="green", lw=1, alpha=0.6)
    ax[0, 0].set_title("Accuracy ladder (test AUC)\nbiggest drop = the culprit",
                       fontsize=11, fontweight="bold")
    for i, v in enumerate(vals):
        ax[0, 0].text(v + 0.002, len(names) - 1 - i, f"{v:.3f}", va="center", fontsize=8)

    # (0,1) kernel K heatmap (sorted by label)
    im1 = ax[0, 1].imshow(Ks[np.ix_(order, order)], cmap="viridis", aspect="auto")
    ax[0, 1].set_title(f"Checkpoint 1: kernel K (sorted by label)\n"
                       f"align(K,y)={k_align_y:.3f}  within-between={within-between:+.3f}")
    ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([]); fig.colorbar(im1, ax=ax[0, 1])

    # (0,2) embedding Gram Ghat heatmap (same order)
    im2 = ax[0, 2].imshow(Gs[np.ix_(order, order)], cmap="viridis", aspect="auto")
    ax[0, 2].set_title(f"Checkpoint 2: embedding Gram Ghat\n"
                       f"align(Ghat,K)={g_align_K:.3f}  corr={corr:.3f}")
    ax[0, 2].set_xticks([]); ax[0, 2].set_yticks([]); fig.colorbar(im2, ax=ax[0, 2])

    # (1,0) scatter Ghat vs K
    samp = rng.choice(len(tri[0]), min(4000, len(tri[0])), replace=False)
    ax[1, 0].scatter(Ks[tri][samp], Gs[tri][samp], s=3, alpha=0.25)
    ax[1, 0].set_xlabel("kernel K(i,j)"); ax[1, 0].set_ylabel("embedding cos(i,j)")
    ax[1, 0].set_title(f"Preservation scatter (corr={corr:.3f})\n"
                       "on the diagonal = perfect copy")
    ax[1, 0].grid(alpha=0.3)

    # (1,1) neighbourhood recall@k
    ks = list(recalls.keys())
    ax[1, 1].plot(ks, [recalls[k] for k in ks], "o-", color="tab:purple")
    ax[1, 1].set_ylim(0, 1); ax[1, 1].set_xlabel("k"); ax[1, 1].set_ylabel("recall@k")
    ax[1, 1].set_title("Checkpoint 2: neighbour preservation\n"
                       "how many K-neighbours phi keeps")
    ax[1, 1].grid(alpha=0.3)

    # (1,2) contrastive loss curve (did it even train?)
    ax[1, 2].plot(range(1, len(con_curve) + 1), con_curve, color="tab:orange")
    ax[1, 2].set_xlabel("epoch"); ax[1, 2].set_ylabel("contrastive loss")
    ax[1, 2].set_title(f"Contrastive loss\n{con_curve[0]:.3f} -> {con_curve[-1]:.3f}")
    ax[1, 2].grid(alpha=0.3)

    # (2,0) per-dimension std (collapse)
    ax[2, 0].bar(range(len(dim_std)), np.sort(dim_std)[::-1], color="tab:blue")
    ax[2, 0].axhline(1e-3, ls=":", c="red", label="dead threshold")
    ax[2, 0].set_xlabel("embedding dim (sorted)"); ax[2, 0].set_ylabel("std")
    ax[2, 0].set_title(f"Collapse: per-dim spread\n"
                       f"eff.rank={erank:.1f}/{args.embedding_dim}, dead={dead}")
    ax[2, 0].legend(fontsize=8)

    # (2,1) 2D PCA of phi coloured by label
    Zc = phi_te - phi_te.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    proj = Zc @ Vt[:2].T
    for c in range(C):
        m = ds.y_test == c
        ax[2, 1].scatter(proj[m, 0], proj[m, 1], s=5, alpha=0.4, label=f"class {c}")
    ax[2, 1].set_title("Checkpoint 3: phi(test) in 2D (PCA)\nclasses should separate")
    ax[2, 1].legend(fontsize=8, markerscale=2)

    # (2,2) probes side by side
    probe_names = ["phi", "leaf-1hot", "raw X"]
    probe_vals = [probe_phi["auc"], probe_leaf["auc"], probe_raw["auc"]]
    ax[2, 2].bar(probe_names, probe_vals, color=["#d62728", "#7f7f7f", "#7f7f7f"])
    ax[2, 2].axhline(tree_auc, ls="--", c="green", label=f"tree {tree_auc:.3f}")
    ax[2, 2].set_ylim(0.5, max(0.75, tree_auc + 0.02))
    ax[2, 2].set_title("Checkpoint 3: linear-probe AUC\nis phi as good as leaf-onehot?")
    ax[2, 2].legend(fontsize=8)
    for i, v in enumerate(probe_vals):
        ax[2, 2].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle(f"TKCE diagnosis — {ds.name} [{args.kernel} kernel] | "
                 f"full head AUC {full['auc']:.3f} "
                 f"vs tree {tree_auc:.3f} | biggest drop: {worst[0]}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = os.path.join(args.out, f"diag_{ds.name}_{args.kernel}.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"[diagnosis] figure  -> {png}")
    print(f"[diagnosis] summary -> {args.out}/diag_{ds.name}_{args.kernel}.json (+ _ladder.csv)")


if __name__ == "__main__":
    main()
