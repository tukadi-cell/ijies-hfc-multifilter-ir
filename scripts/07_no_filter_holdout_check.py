# -*- coding: utf-8 -*-
"""
Evaluasi holdout kontrol TANPA FILTER pada SEMUA ZIP eksternal: (8) dan (2).

- TIDAK melatih ulang. Memakai model final yang sudah disimpan oleh notebook no-filter
  (no_filter_final_model.keras) dan referensi background yang sama (61 frame independen
  dari no_filter_background_manifest.csv). Hasil CV (58.11%) tidak berubah.
- Langkah 1 memeriksa reproduksi: ZIP (8) harus kembali memberi 54.24% (128/236).
- Lalu ZIP (2) dievaluasi dengan cara yang sama, dan hasil dilaporkan per ZIP,
  per sesi, dan gabungan, disertai pemeriksaan kebocoran:
    * nomor sesi holdout tidak boleh sama dengan sesi pengembangan (2-8),
    * isi frame tidak boleh identik dengan frame pengembangan,
    * duplikat antar-ZIP holdout dicatat dan dihitung sekali.

Keluaran di OUTPUT_DIR/holdout_all_zips/:
  inventory_<zip>.csv, predictions_all_holdout.csv, summary_by_zip.csv,
  summary_by_session.csv, summary_all.json, checks.txt
"""
import os, re, json, shutil, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# ===================== PENGATURAN =====================
DATA_DIR = Path(r"C:\1. Tukadi_ITS\2. Kumpulan data jurnal2\.data_5(m)\.data_3(m)")
HOLDOUT_ZIPS = {
    "ZIP8": DATA_DIR / "X_Exsternal  (8)_NO_FILTER.zip",   # holdout yang sudah dilaporkan
    "ZIP2": DATA_DIR / "X_Exsternal  (2)_NO_FILTER.zip",   # holdout tambahan
}
NOFILTER_OUT = Path(r"C:\Users\tukadi\OneDrive\Documents\ChatGPT\New project\ijies_no_filter_outputs")
MODEL_PATH = NOFILTER_OUT / "no_filter_final_model.keras"
BACKGROUND_MANIFEST = NOFILTER_OUT / "no_filter_background_manifest.csv"
SOURCE_MANIFEST = NOFILTER_OUT / "no_filter_source_manifest.csv"
EXPECTED_ZIP8_ACCURACY = 0.5424         # nilai di naskah (128/236)
LABEL_TO_ID = {"Clean air": 0, "HFC": 1}
OUT = NOFILTER_OUT / "holdout_all_zips"
# ======================================================

checks = []


def note(s):
    print(s, flush=True); checks.append(s)


def session_number(path):
    m = re.search(r"\((\d+)\)", Path(path).stem)
    return int(m.group(1)) if m else None


def frame_number(path):
    m = re.search(r"_frame_(\d+)", Path(path).stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def read_frame(path):
    a = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    if a.shape == (24, 32):
        return a
    if a.size == 768:
        return a.reshape(24, 32)
    raise ValueError(f"Invalid frame shape {a.shape}: {path}")


def value_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def extract(zip_path, name):
    dest = OUT / f"_extracted_{name}"
    if not dest.exists():
        dest.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                target = (dest / info.filename).resolve()
                if dest.resolve() not in target.parents and target != dest.resolve():
                    raise ValueError(f"Unsafe ZIP member: {info.filename}")
            z.extractall(dest)
    return dest


def find_class_dir(root, wanted):
    w = wanted.casefold().replace(" ", "")
    for p in Path(root).rglob("*"):
        if p.is_dir() and p.name.casefold().replace(" ", "") == w:
            return p
    return None


def metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)), "macro_precision": float(pr),
            "macro_recall": float(rc), "macro_f1": float(f1), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "hfc_recall": float(tp / max(tp + fn, 1)), "false_positive_rate": float(fp / max(fp + tn, 1))}


def main():
    import tensorflow as tf
    OUT.mkdir(parents=True, exist_ok=True)

    # Model dan referensi background yang SAMA dengan notebook no-filter
    model = tf.keras.models.load_model(MODEL_PATH)
    bm = pd.read_csv(BACKGROUND_MANIFEST)
    sel = bm[bm["selected_for_reference"].astype(str).str.lower() == "true"]
    background = np.stack([read_frame(p) for p in sel["path"]]).mean(axis=0).astype(np.float32)
    note(f"Background reference: {len(sel)} independent frames (expected 61)")
    dev = pd.read_csv(SOURCE_MANIFEST)
    dev_sessions, dev_hashes = set(dev["session"]), set(dev["value_sha256"])
    note(f"Development sessions: {sorted(dev_sessions)}")

    rows = []
    for name, zp in HOLDOUT_ZIPS.items():
        if not zp.exists():
            note(f"[{name}] TIDAK DITEMUKAN: {zp}"); continue
        root = extract(zp, name)
        inv = []
        for cls, label in LABEL_TO_ID.items():
            d = find_class_dir(root, cls)
            if d is None:
                note(f"[{name}] folder '{cls}' tidak ditemukan"); continue
            for p in sorted(d.glob("*.csv")):
                a = read_frame(p)
                inv.append({"zip": name, "path": str(p), "class_name": cls, "true": label,
                            "session": session_number(p), "frame": frame_number(p), "value_sha256": value_hash(a)})
                inv[-1]["_array"] = a
        df = pd.DataFrame(inv)
        if df.empty:
            continue
        counts = pd.crosstab(df["session"], df["class_name"])
        note(f"\n[{name}] {len(df)} frame; sesi: {sorted(df['session'].dropna().astype(int).unique().tolist())}")
        note(counts.to_string())
        overlap_s = set(df["session"]) & dev_sessions
        overlap_v = int(df["value_sha256"].isin(dev_hashes).sum())
        note(f"[{name}] sesi yang sama dengan pengembangan: {sorted(overlap_s) or 'tidak ada'}")
        note(f"[{name}] frame identik dengan data pengembangan: {overlap_v}")
        df.drop(columns="_array").to_csv(OUT / f"inventory_{name}.csv", index=False)
        df["leak_session"] = df["session"].isin(dev_sessions)
        df["leak_content"] = df["value_sha256"].isin(dev_hashes)
        rows.append(df)
    frames = pd.concat([r for r in rows if isinstance(r, pd.DataFrame)], ignore_index=True)

    # Duplikat antar-ZIP holdout
    dup = frames.duplicated("value_sha256", keep="first")
    note(f"\nDuplikat isi frame antar/inner holdout ZIP: {int(dup.sum())} (dihitung sekali)")
    frames["duplicate_holdout"] = dup

    x = (np.stack(frames["_array"].values) - background).astype(np.float32)[..., None]
    prob = model.predict(x, verbose=0)
    frames["pred"] = prob.argmax(axis=1)
    frames["p_hfc"] = prob[:, 1]
    frames["correct"] = frames["pred"] == frames["true"]
    frames.drop(columns="_array").to_csv(OUT / "predictions_all_holdout.csv", index=False)

    usable = frames[~frames["leak_session"] & ~frames["leak_content"] & ~frames["duplicate_holdout"]]
    by_zip = pd.DataFrame([{"zip": z, **metrics(g["true"].values, g["pred"].values)} for z, g in usable.groupby("zip")])
    by_ses = pd.DataFrame([{"zip": z, "session": int(s), **metrics(g["true"].values, g["pred"].values)}
                           for (z, s), g in usable.groupby(["zip", "session"])])
    overall = metrics(usable["true"].values, usable["pred"].values)
    by_zip.to_csv(OUT / "summary_by_zip.csv", index=False)
    by_ses.to_csv(OUT / "summary_by_session.csv", index=False)
    (OUT / "summary_all.json").write_text(json.dumps({k: float(v) if isinstance(v, (np.floating, float)) else v
                                                       for k, v in overall.items()}, indent=2), encoding="utf-8")

    if "ZIP8" in set(by_zip["zip"]):
        acc8 = float(by_zip.loc[by_zip["zip"] == "ZIP8", "accuracy"].iloc[0])
        ok = abs(acc8 - EXPECTED_ZIP8_ACCURACY) < 0.0005
        note(f"\nReproduksi ZIP (8): {acc8:.4f} -> {'COCOK dengan naskah' if ok else 'TIDAK COCOK, periksa model/background'}")
    pd.set_option("display.width", 200)
    print("\n=== Per ZIP ===\n", by_zip.round(4).to_string(index=False))
    print("\n=== Per sesi ===\n", by_ses.round(4).to_string(index=False))
    print("\n=== Gabungan (tanpa frame bocor/duplikat) ===\n", {k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()})
    (OUT / "checks.txt").write_text("\n".join(checks), encoding="utf-8")
    print("\nSelesai. Hasil di:", OUT)


if __name__ == "__main__":
    main()
