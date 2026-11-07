# -*- coding: utf-8 -*-
"""
Analisis tambahan untuk revisi mayor IJIES 20267306 (memakai data dan fold yang sama).

  A. Ablasi region & registrasi dengan model NON-CNN (random forest) dan model ringkasan
     intensitas region sederhana                                         [Reviewer 2-4]
  B. Confusion matrix per sesi (CNN) dan selisih berpasangan per sesi untuk ablasi
     (CNN dari artefak + RF dari bagian A), dengan bootstrap per sesi     [Reviewer 2-5]
  C. Bias early stopping: CNN dengan jadwal epoch tetap dan dengan inner split per sesi
                                                                          [Reviewer 1-5]
  D. Augmentasi SEBELUM vs SESUDAH pengurangan background (jadwal epoch tetap)
                                                                          [Reviewer 1-2]
  E. Kurva sensitivitas-spesifisitas pada data pengembangan (out-of-fold), ambang dibekukan,
     lalu diterapkan sekali pada holdout                                  [Reviewer 2-6]

Fold diambil langsung dari source_fold_manifest.csv (artefak pipeline utama), sehingga
pembagian sesi identik dengan naskah. Setiap bagian bisa dimatikan di PENGATURAN.
"""
import os, re, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# ===================== PENGATURAN =====================
MASTER_OUT = Path(r"C:\Users\tukadi\OneDrive\Documents\ChatGPT\New project\ijies_master_outputs")
DATA_PARENT = Path(r"C:\1. Tukadi_ITS")             # berisi X_learning5 dan X_testing5
LEARNING_ROOT = DATA_PARENT / "X_learning5"
HOLDOUT_ROOT = DATA_PARENT / "X_testing5"
OUT = MASTER_OUT / "reviewer_round2_analyses"
RUN = {"A": True, "B": True, "C": True, "D": True, "E": True}
SEED = 42
FIXED_EPOCHS = 4            # aturan training final di naskah (median epoch terbaik)
MAX_EPOCHS, PATIENCE, BATCH = 50, 30, 32
N_BOOT = 5000
# Mask region (indeks 0-based, ujung atas eksklusif) = baris 3-11/14-22, kolom 3-14/19-30 (1-based)
REGIONS = {"F1": (2, 11, 2, 14), "F2": (2, 11, 18, 30), "F3": (13, 22, 2, 14), "F4": (13, 22, 18, 30)}
# ======================================================

rng_global = np.random.default_rng(SEED)
log_lines = []


def log(s=""):
    print(s, flush=True); log_lines.append(str(s))


# ---------------------------------------------------------------- data
def read_frame(path):
    p = str(path)
    head = open(p, encoding="utf-8", errors="ignore").read(2000)
    a = np.loadtxt(p, delimiter=",") if "," in head else np.loadtxt(p)
    return a.reshape(24, 32).astype(np.float32)


def resolve(p):
    p = Path(str(p).replace("\\", "/"))
    if p.exists():
        return p
    parts = p.parts
    for i, part in enumerate(parts):
        if part.lower().startswith("x_learning") or part.lower().startswith("x_testing"):
            q = DATA_PARENT.joinpath(*parts[i:])
            if q.exists():
                return q
    raise FileNotFoundError(p)


def number(p):
    m = re.search(r"\((\d+)\)", Path(p).name)
    return int(m.group(1)) if m else 0


def load_all():
    man = pd.read_csv(MASTER_OUT / "source_fold_manifest.csv")
    bg_files = sorted((LEARNING_ROOT / "Background").glob("*.csv"))
    background = np.mean([read_frame(p) for p in bg_files], axis=0).astype(np.float32)
    raw = np.stack([read_frame(resolve(p)) for p in man["path"]])
    y = man["label"].to_numpy().astype(int)
    sess = man["session"].to_numpy()
    fold = man["validation_fold"].to_numpy()
    hold = []
    for label, cls in ((0, "Clean Air"), (1, "HFC")):
        for p in sorted((HOLDOUT_ROOT / cls).glob("*.csv"), key=number):
            hold.append((read_frame(p), label, f"T{(number(p) - 1) // 100 + 8}"))
    hraw = np.stack([h[0] for h in hold]); hy = np.array([h[1] for h in hold]); hs = np.array([h[2] for h in hold])
    log(f"Data: {len(y)} development frames, {len(np.unique(sess))} sessions, folds {sorted(np.unique(fold))}; "
        f"holdout {len(hy)}; background {len(bg_files)} frames")
    return raw, y, sess, fold, background, hraw, hy, hs


# ---------------------------------------------------------------- augmentation (6 deterministic derivatives)
def derivatives(frame, clip=True):
    h, w = frame.shape
    out = []
    for ang in (5, -5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out.append(cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    for off in (15, -15):
        out.append(np.clip(frame + off, 0, 255) if clip else frame + off)
    zh, zw = int(round(h / 1.1)), int(round(w / 1.1))
    y0, x0 = (h - zh) // 2, (w - zw) // 2
    out.append(cv2.resize(frame[y0:y0 + zh, x0:x0 + zw], (w, h), interpolation=cv2.INTER_LINEAR))
    out.append(cv2.GaussianBlur(frame, (3, 3), 0))
    return [frame] + [o.astype(np.float32) for o in out]


def train_set(raw, y, idx, background, order="before"):
    """order='before': augment raw 8-bit frames, then subtract background (current pipeline);
       order='after' : subtract background first, then augment the temperature-change image."""
    X, Y = [], []
    for i in idx:
        if order == "before":
            ds = [d - background for d in derivatives(raw[i])]
        else:
            ds = derivatives(raw[i] - background, clip=False)
        X += ds; Y += [y[i]] * len(ds)
    return np.stack(X).astype(np.float32), np.array(Y)


# ---------------------------------------------------------------- region features
def region_mask(names, shift=(0, 0)):
    m = np.zeros((24, 32), bool)
    dr, dc = shift
    for n in names:
        r0, r1, c0, c1 = REGIONS[n]
        m[max(r0 + dr, 0):min(r1 + dr, 24), max(c0 + dc, 0):min(c1 + dc, 32)] = True
    return m


def region_vectors(X, names, shift=(0, 0)):
    """concatenate region pixels in fixed F1..F4 order; shift moves the read window"""
    dr, dc = shift
    parts = []
    for n in names:
        r0, r1, c0, c1 = REGIONS[n]
        parts.append(X[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc].reshape(len(X), -1))
    return np.concatenate(parts, axis=1)


def region_summary(X, shift=(0, 0)):
    feats = []
    dr, dc = shift
    for n in ("F1", "F2", "F3", "F4"):
        r0, r1, c0, c1 = REGIONS[n]
        b = X[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc].reshape(len(X), -1)
        feats += [b.mean(1), b.std(1), b.min(1), b.max(1)]
    return np.stack(feats, axis=1)


PERMUTATIONS = {"Horizontal swap": ["F2", "F1", "F4", "F3"], "Vertical swap": ["F3", "F4", "F1", "F2"],
                "Diagonal swap": ["F4", "F3", "F2", "F1"], "Cyclic shift": ["F4", "F1", "F2", "F3"]}


def permute_regions(X, order):
    Z = X.copy()
    for dst, src in zip(("F1", "F2", "F3", "F4"), order):
        r0, r1, c0, c1 = REGIONS[dst]; s0, s1, t0, t1 = REGIONS[src]
        Z[:, r0:r1, c0:c1] = X[:, s0:s1, t0:t1]
    return Z


# ---------------------------------------------------------------- models
def rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def logreg():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=SEED))])


def build_cnn(tf):
    L = tf.keras.layers
    m = tf.keras.Sequential([L.Input((24, 32, 1))])
    for f in (32, 64, 128):
        m.add(L.Conv2D(f, 3, padding="same", activation="relu"))
        m.add(L.BatchNormalization())
        m.add(L.MaxPooling2D(2))
    m.add(L.Flatten()); m.add(L.Dense(128, activation="relu")); m.add(L.Dropout(0.5)); m.add(L.Dense(2, activation="softmax"))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def fit_cnn(tf, Xtr, ytr, seed, epochs, Xval=None, yval=None):
    tf.keras.utils.set_random_seed(seed)
    m = build_cnn(tf)
    yt = tf.keras.utils.to_categorical(ytr, 2)
    if Xval is None:
        m.fit(Xtr[..., None], yt, epochs=epochs, batch_size=BATCH, shuffle=True, verbose=0)
        return m, epochs
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    h = m.fit(Xtr[..., None], yt, epochs=MAX_EPOCHS, batch_size=BATCH, shuffle=True, verbose=0,
              validation_data=(Xval[..., None], tf.keras.utils.to_categorical(yval, 2)), callbacks=[cb])
    return m, int(np.argmin(h.history["val_loss"]) + 1)


# ---------------------------------------------------------------- statistics
def acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def session_acc(df, col="pred"):
    return df.groupby("session").apply(lambda g: acc(g["true"], g[col])).sort_index()


def paired_boot(a, b):
    """session-level paired bootstrap of mean(b-a), a/b = per-session accuracy series"""
    d = (b - a).to_numpy()
    r = np.random.default_rng(SEED)
    boots = [d[r.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


# ================================================================ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, y, sess, fold, bg, hraw, hy, hs = load_all()
    delta = raw - bg
    folds = sorted(np.unique(fold))

    # ------------------------------------------------ A. RF / summary-model ablation
    rf_preds = []
    if RUN["A"]:
        log("\n=== A. Region ablation and registration with non-CNN models ===")
        conds = [("Spatial reference (full frame)", "full", None, (0, 0)),
                 ("Four-region baseline", "regions", ["F1", "F2", "F3", "F4"], (0, 0))]
        conds += [(f"Retain {n} only", "regions", [n], (0, 0)) for n in REGIONS]
        conds += [(f"Remove {n}", "regions", [m for m in REGIONS if m != n], (0, 0)) for n in REGIONS]
        for model_name, make in (("Random forest", rf),):
            for name, kind, names, sh in conds:
                t0 = time.time()
                for f in folds:
                    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
                    Xtr, ytr = train_set(raw, y, tr, bg)
                    if kind == "full":
                        A, B = Xtr.reshape(len(Xtr), -1), delta[va].reshape(len(va), -1)
                    else:
                        A, B = region_vectors(Xtr, names), region_vectors(delta[va], names)
                    m = make().fit(A, ytr)
                    for i, p in zip(va, m.predict(B)):
                        rf_preds.append({"model": model_name, "condition": name, "fold": f, "source_index": i,
                                         "session": sess[i], "true": y[i], "pred": int(p)})
                log(f"  {model_name:14s} {name:32s} done ({time.time() - t0:.0f} s)")
            # frozen permutation and registration shifts: train once on four-region per fold
            for f in folds:
                tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
                Xtr, ytr = train_set(raw, y, tr, bg)
                m = make().fit(region_vectors(Xtr, list(REGIONS)), ytr)
                for pname, order in PERMUTATIONS.items():
                    pv = m.predict(region_vectors(permute_regions(delta[va], order), list(REGIONS)))
                    rf_preds += [{"model": model_name, "condition": f"Frozen permutation: {pname}", "fold": f,
                                  "source_index": i, "session": sess[i], "true": y[i], "pred": int(p)} for i, p in zip(va, pv)]
                for sh in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0)]:
                    pv = m.predict(region_vectors(delta[va], list(REGIONS), sh))
                    rf_preds += [{"model": model_name, "condition": f"Registration shift (row {sh[0]:+d}, col {sh[1]:+d})",
                                  "fold": f, "source_index": i, "session": sess[i], "true": y[i], "pred": int(p)} for i, p in zip(va, pv)]
        # simple region-summary models (no pixel layout)
        for model_name, make, feat in (("Region-mean logistic regression", logreg, "mean"),
                                       ("Region-summary random forest", rf, "summary")):
            for f in folds:
                tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
                Xtr, ytr = train_set(raw, y, tr, bg)
                A, B = region_summary(Xtr), region_summary(delta[va])
                if feat == "mean":
                    A, B = A[:, 0::4], B[:, 0::4]
                m = make().fit(A, ytr)
                rf_preds += [{"model": model_name, "condition": "Four-region summary", "fold": f, "source_index": i,
                              "session": sess[i], "true": y[i], "pred": int(p)} for i, p in zip(va, m.predict(B))]
            log(f"  {model_name} done")
        rfp = pd.DataFrame(rf_preds); rfp.to_csv(OUT / "A_noncnn_ablation_predictions.csv", index=False)
        summ = rfp.groupby(["model", "condition"]).apply(lambda g: pd.Series({
            "pooled_accuracy_%": 100 * acc(g["true"], g["pred"]),
            "fold_mean_%": 100 * g.groupby("fold").apply(lambda h: acc(h["true"], h["pred"])).mean(),
            "fold_sd_%": 100 * g.groupby("fold").apply(lambda h: acc(h["true"], h["pred"])).std(ddof=1)})).reset_index()
        summ.to_csv(OUT / "A_noncnn_ablation_summary.csv", index=False)
        log(summ.round(2).to_string(index=False))

    # ------------------------------------------------ B. per-session confusion + paired ablation differences
    if RUN["B"]:
        log("\n=== B. Session-level uncertainty ===")
        cnn = pd.read_csv(MASTER_OUT / "cnn_source_predictions.csv")
        cm = cnn.groupby("session").apply(lambda g: pd.Series({
            "n": len(g), "TN": int(((g.true == 0) & (g.pred == 0)).sum()), "FP": int(((g.true == 0) & (g.pred == 1)).sum()),
            "FN": int(((g.true == 1) & (g.pred == 0)).sum()), "TP": int(((g.true == 1) & (g.pred == 1)).sum()),
            "accuracy_%": 100 * acc(g.true, g.pred)})).reset_index()
        cm.to_csv(OUT / "B_cnn_confusion_by_session.csv", index=False)
        log("CNN confusion by session:\n" + cm.to_string(index=False))
        rows = []
        t3 = pd.read_csv(MASTER_OUT / "table3_source_predictions.csv")
        smap = cnn.set_index("source_index")["session"]
        t3["session"] = t3["source_index"].map(smap)
        t3["model"] = "CNN"; t3["condition"] = t3["condition"].astype(str)
        frames = [t3[["model", "condition", "session", "true", "pred"]]]
        if RUN["A"]:
            frames.append(rfp[["model", "condition", "session", "true", "pred"]])
        allp = pd.concat(frames, ignore_index=True)
        for model, g in allp.groupby("model"):
            base_name = [c for c in g["condition"].unique() if "four" in c.lower() and ("baseline" in c.lower() or "region" in c.lower()) and "summary" not in c.lower()]
            if not base_name:
                continue
            base = session_acc(g[g["condition"] == base_name[0]])
            for cond, gc in g.groupby("condition"):
                if cond == base_name[0]:
                    continue
                s = session_acc(gc).reindex(base.index)
                mdiff, lo, hi = paired_boot(base, s)
                rows.append({"model": model, "condition": cond, "baseline": base_name[0],
                             "mean_session_diff_pp": mdiff, "ci95_low": lo, "ci95_high": hi})
        pdiff = pd.DataFrame(rows); pdiff.to_csv(OUT / "B_paired_session_differences.csv", index=False)
        log("Paired session-level differences vs four-region baseline (pp, 95% CI):\n" + pdiff.round(2).to_string(index=False))

    # ------------------------------------------------ C/D. CNN retraining analyses
    if RUN["C"] or RUN["D"]:
        import tensorflow as tf
        res = []

        def run_cv(label, order="before", mode="fixed"):
            preds = []
            epochs_used = []
            for k, f in enumerate(folds):
                tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
                if mode == "inner":
                    tr_sess = np.unique(sess[tr])
                    inner_val_s = tr_sess[np.random.default_rng(SEED + k).integers(len(tr_sess))]
                    itr, iva = tr[sess[tr] != inner_val_s], tr[sess[tr] == inner_val_s]
                    Xtr, ytr = train_set(raw, y, itr, bg, order)
                    _, best = fit_cnn(tf, Xtr, ytr, SEED + k, None, delta[iva], y[iva])
                    Xall, yall = train_set(raw, y, tr, bg, order)
                    m, _ = fit_cnn(tf, Xall, yall, SEED + k, best)
                    epochs_used.append(best)
                else:
                    Xtr, ytr = train_set(raw, y, tr, bg, order)
                    m, _ = fit_cnn(tf, Xtr, ytr, SEED + k, FIXED_EPOCHS)
                    epochs_used.append(FIXED_EPOCHS)
                p = m.predict(delta[va][..., None], verbose=0).argmax(1)
                preds += [{"analysis": label, "fold": f, "source_index": i, "session": sess[i], "true": y[i], "pred": int(q)}
                          for i, q in zip(va, p)]
                log(f"  {label}: fold {f} done (epochs {epochs_used[-1]})")
            d = pd.DataFrame(preds)
            res.append({"analysis": label, "pooled_accuracy_%": 100 * acc(d.true, d.pred),
                        "fold_mean_%": 100 * d.groupby("fold").apply(lambda g: acc(g.true, g.pred)).mean(),
                        "fold_sd_%": 100 * d.groupby("fold").apply(lambda g: acc(g.true, g.pred)).std(ddof=1),
                        "epochs": ",".join(map(str, epochs_used))})
            return d

        allc = []
        if RUN["C"]:
            log("\n=== C. Early-stopping bias ===")
            allc.append(run_cv("C1 fixed 4-epoch schedule (no early stopping)", "before", "fixed"))
            allc.append(run_cv("C2 inner session split for epoch selection", "before", "inner"))
        if RUN["D"]:
            log("\n=== D. Augmentation order (fixed 4-epoch schedule) ===")
            if not RUN["C"]:
                allc.append(run_cv("C1 fixed 4-epoch schedule (no early stopping)", "before", "fixed"))
            allc.append(run_cv("D1 augmentation after background subtraction", "after", "fixed"))
        pd.concat(allc).to_csv(OUT / "CD_cnn_retraining_predictions.csv", index=False)
        cd = pd.DataFrame(res); cd.to_csv(OUT / "CD_cnn_retraining_summary.csv", index=False)
        log("Reported CV in manuscript (early stopping on validation folds): 80.29% pooled")
        log(cd.round(2).to_string(index=False))

    # ------------------------------------------------ E. operating threshold
    if RUN["E"]:
        from sklearn.metrics import roc_curve, roc_auc_score
        log("\n=== E. Sensitivity-specificity trade-off and frozen threshold ===")
        cnn = pd.read_csv(MASTER_OUT / "cnn_source_predictions.csv")
        ho = pd.read_csv(MASTER_OUT / "holdout_source_predictions.csv")
        fpr, tpr, thr = roc_curve(cnn["true"], cnn["p_hfc"])
        auc_dev = roc_auc_score(cnn["true"], cnn["p_hfc"])
        roc = pd.DataFrame({"threshold": thr, "sensitivity": tpr, "specificity": 1 - fpr})
        roc.to_csv(OUT / "E_dev_roc.csv", index=False)
        j = roc.iloc[1:].assign(J=lambda r: r.sensitivity + r.specificity - 1).sort_values("J", ascending=False).iloc[0]
        spec90 = roc[roc.specificity >= 0.90].sort_values("sensitivity", ascending=False).iloc[0]
        rows = []
        for name, t in (("Default 0.50", 0.5), ("Youden-optimal on development", float(j.threshold)),
                        ("Specificity >= 0.90 on development", float(spec90.threshold))):
            for dname, d in (("Development (out-of-fold)", cnn), ("Holdout (frozen threshold)", ho)):
                p = (d["p_hfc"] >= t).astype(int)
                tp = int(((p == 1) & (d.true == 1)).sum()); tn = int(((p == 0) & (d.true == 0)).sum())
                fp = int(((p == 1) & (d.true == 0)).sum()); fn = int(((p == 0) & (d.true == 1)).sum())
                rows.append({"rule": name, "threshold": t, "data": dname, "sensitivity_%": 100 * tp / max(tp + fn, 1),
                             "specificity_%": 100 * tn / max(tn + fp, 1), "accuracy_%": 100 * (tp + tn) / len(d)})
        th = pd.DataFrame(rows); th.to_csv(OUT / "E_threshold_results.csv", index=False)
        log(f"Development AUC (out-of-fold): {auc_dev:.4f}; holdout AUC: {roc_auc_score(ho['true'], ho['p_hfc']):.4f}")
        log(th.round(3).to_string(index=False))

    (OUT / "analysis_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nSelesai. Semua hasil di: {OUT}")


if __name__ == "__main__":
    main()
