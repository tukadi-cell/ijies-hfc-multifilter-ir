# -*- coding: utf-8 -*-
"""
Perbandingan ukuran model dan latensi inferensi:
compact CNN vs classifier dengan akurasi lebih tinggi di Tabel 13.

Semua model diukur di komputer yang sama, dengan input yang sama
(frame 24x32 hasil pengurangan background).

Keluaran (OUTPUT_DIR):
  - resource_comparison.csv
  - resource_comparison.png
"""
import os, glob, re, time, pickle, tempfile
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ===================== PENGATURAN =====================
DATA_ROOT = r"C:\1. Tukadi_ITS\X_learning5"          # berisi Clean Air, HFC, Background
CNN_MODEL = r"C:\Users\tukadi\OneDrive\Documents\ChatGPT\New project\ijies_master_outputs\final_model_fixed_epoch.keras"
OUTPUT_DIR = r"C:\1. Tukadi_ITS\hasil_resource"
SEED = 42
N_SINGLE = 500        # jumlah prediksi satu-frame untuk mengukur latensi
WARMUP = 20


def comparator_models():
    """Ke-14 classifier pembanding dengan hyperparameter PERSIS seperti Tabel 13
    (diambil dari notebook pipeline utama dan notebook 'classifier All')."""
    from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
                                  GradientBoostingClassifier, AdaBoostClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _scaled(m):
        return Pipeline([("scale", StandardScaler()), ("model", m)])
    return {
        "Extra trees": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Random forest": RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Histogram gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
        "RBF SVM": _scaled(SVC(C=1, kernel="rbf", gamma="scale")),
        "MLP (sklearn)": _scaled(MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True, random_state=SEED)),
        "Decision tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        "Linear SVM": _scaled(SVC(C=1, kernel="linear")),
        "K-nearest neighbors": _scaled(KNeighborsClassifier(n_neighbors=5)),
        "Logistic regression": _scaled(LogisticRegression(max_iter=2000, random_state=SEED)),
        "Linear discriminant analysis": _scaled(LinearDiscriminantAnalysis()),
        "Ridge classifier": _scaled(RidgeClassifier()),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.10, max_depth=2, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=SEED),
        "Gaussian naive Bayes": _scaled(GaussianNB()),
    }
# ======================================================


def read_frame(path):
    a = np.loadtxt(path, delimiter=",") if open(path).read(2000).count(",") else np.loadtxt(path)
    return a.reshape(24, 32).astype(np.float32)


def number(p):
    m = re.search(r"\((\d+)\)", os.path.basename(p))
    return int(m.group(1)) if m else 0


def list_csv(folder):
    return sorted(glob.glob(os.path.join(folder, "*.csv")), key=number)


def augment(raw):
    """Enam turunan deterministik seperti di pipeline utama (Section 2.3)."""
    h, w = raw.shape
    out = []
    for ang in (5, -5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out.append(cv2.warpAffine(raw, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    for off in (15, -15):
        out.append(np.clip(raw + off, 0, 255))
    zh, zw = int(round(h / 1.1)), int(round(w / 1.1))
    y0, x0 = (h - zh) // 2, (w - zw) // 2
    out.append(cv2.resize(raw[y0:y0 + zh, x0:x0 + zw], (w, h), interpolation=cv2.INTER_LINEAR))
    out.append(cv2.GaussianBlur(raw, (3, 3), 0))
    return out


def load_training_data():
    bg = np.mean([read_frame(p) for p in list_csv(os.path.join(DATA_ROOT, "Background"))], axis=0)
    X, y = [], []
    for label, cls in ((0, "Clean Air"), (1, "HFC")):
        for p in list_csv(os.path.join(DATA_ROOT, cls)):
            raw = read_frame(p)
            for f in [raw] + augment(raw):
                X.append(f - bg)
                y.append(label)
    return np.asarray(X, np.float32), np.asarray(y)


def median_latency_ms(predict_one, x):
    for _ in range(WARMUP):
        predict_one(x)
    t = []
    for _ in range(N_SINGLE):
        t0 = time.perf_counter()
        predict_one(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def make_tflite_runner(content, tf):
    """Coba LiteRT (pengganti resmi tf.lite.Interpreter), lalu tf.lite. Kembalikan None bila gagal."""
    candidates = []
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRT
        candidates.append(("ai_edge_litert", LiteRT))
    except Exception:
        pass
    candidates.append(("tf.lite", tf.lite.Interpreter))
    for label, Interp in candidates:
        try:
            interp = Interp(model_content=content)
            interp.allocate_tensors()
            i_in = interp.get_input_details()[0]["index"]
            i_out = interp.get_output_details()[0]["index"]

            def run(a, interp=interp, i_in=i_in, i_out=i_out):
                interp.set_tensor(i_in, np.asarray(a, np.float32))
                interp.invoke()
                return interp.get_tensor(i_out)
            run(np.zeros((1, 24, 32, 1), np.float32))
            print("  Interpreter TFLite:", label)
            return run
        except Exception as e:
            print(f"  Interpreter {label} gagal: {type(e).__name__}: {e}")
    return None


def sklearn_details(name, model):
    m = model[-1] if hasattr(model, "steps") else model
    if hasattr(m, "estimators_"):
        ests = list(np.ravel(m.estimators_))
        nodes = sum(e.tree_.node_count for e in ests if hasattr(e, "tree_"))
        return f"{len(ests)} trees, {nodes:,} nodes"
    if hasattr(m, "tree_"):
        return f"1 tree, {m.tree_.node_count:,} nodes"
    if hasattr(m, "support_vectors_"):
        return f"{m.support_vectors_.shape[0]:,} support vectors"
    if hasattr(m, "n_samples_fit_"):
        return f"stores {m.n_samples_fit_:,} training samples"
    if hasattr(m, "coefs_"):
        return f"{sum(c.size + b.size for c, b in zip(m.coefs_, m.intercepts_)):,} parameters"
    if hasattr(m, "_predictors"):
        return f"{len(m._predictors)} boosting iterations"
    return ""


def main():
    import tensorflow as tf
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("Memuat data training-development dan augmentasi...")
    X, y = load_training_data()
    print(f"  {X.shape[0]} input training, frame {X.shape[1]}x{X.shape[2]}")
    x1 = X[:1]
    rows = []

    # ---------- CNN ----------
    cnn = tf.keras.models.load_model(CNN_MODEL)
    xin = tf.constant(x1[..., None])
    graph_predict = tf.function(lambda a: cnn(a, training=False))   # mode graph, lebih cepat dari eager
    lat_keras = median_latency_ms(graph_predict, xin)
    conv = tf.lite.TFLiteConverter.from_keras_model(cnn)
    tfl32 = conv.convert()
    conv = tf.lite.TFLiteConverter.from_keras_model(cnn)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl8 = conv.convert()

    lat_tfl = np.nan
    runner = make_tflite_runner(tfl8, tf)
    if runner is not None:
        lat_tfl = median_latency_ms(runner, x1[..., None])
    else:
        print("  Catatan: interpreter TFLite tidak tersedia; latensi int8 tidak diukur, ukuran tetap dilaporkan.")
    rows.append({"model": "Compact CNN (Keras)", "size_kB": os.path.getsize(CNN_MODEL) / 1024,
                 "latency_ms_per_frame": lat_keras, "details": f"{cnn.count_params():,} parameters"})
    rows.append({"model": "Compact CNN (TFLite float32)", "size_kB": len(tfl32) / 1024,
                 "latency_ms_per_frame": np.nan, "details": "converted from the final model"})
    rows.append({"model": "Compact CNN (TFLite dynamic-range int8)", "size_kB": len(tfl8) / 1024,
                 "latency_ms_per_frame": lat_tfl, "details": "post-training quantization"})

    # ---------- Classifier pembanding ----------
    Xf = X.reshape(len(X), -1)
    for name, model in comparator_models().items():
        print("Melatih", name, "...")
        t0 = time.perf_counter()
        model.fit(Xf, y)
        fit_s = time.perf_counter() - t0
        est = model[-1] if hasattr(model, "steps") else model
        if hasattr(est, "n_jobs"):
            est.set_params(n_jobs=1)              # latensi satu frame, satu thread
        size_kB = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024
        lat = median_latency_ms(lambda a: model.predict(a), Xf[:1])
        try:
            det = sklearn_details(name, model)
        except Exception:
            det = ""
        rows.append({"model": name, "size_kB": size_kB, "latency_ms_per_frame": lat,
                     "details": f"{det}; fit {fit_s:.1f} s"})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTPUT_DIR, "resource_comparison.csv"), index=False)
    print("\n", res.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ---------- Grafik ----------
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0))
    ypos = np.arange(len(res))
    axes[0].barh(ypos, res["size_kB"], color="#4C72B0")
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(res["model"])
    axes[0].set_xscale("log"); axes[0].set_xlabel("Model size (kB, log scale)")
    lat = res["latency_ms_per_frame"]
    axes[1].barh(ypos, lat.fillna(0), color="#DD8452")
    for yv, v in zip(ypos, lat):
        if np.isnan(v):
            axes[1].text(lat.min(skipna=True) * 1.05, yv, "not measured", va="center", fontsize=7, color="gray")
    axes[1].set_yticks(ypos); axes[1].set_yticklabels([])
    axes[1].set_xscale("log"); axes[1].set_xlabel("Single-frame latency (ms, log scale)")
    for ax in axes:
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "resource_comparison.png"), dpi=300)
    print("\nSelesai. Hasil tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
