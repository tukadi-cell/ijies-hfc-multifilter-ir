# -*- coding: utf-8 -*-
"""
Akurasi CNN final pada holdout 600 frame: float32 vs int8 (dynamic-range quantization).

Langkah:
  1. Memuat model final (final_model_fixed_epoch.keras) dan referensi background
     dari X_learning5/Background (sama dengan pipeline utama).
  2. Mengevaluasi model float32 pada X_testing5. Hasil ini HARUS sama dengan naskah
     (89.67%, 538/600). Jika tidak sama, preprocessing tidak cocok dan hasil int8
     tidak boleh dipakai.
  3. Mengonversi model ke TFLite int8 (dynamic-range) dan mengevaluasinya pada data
     yang sama. Jika interpreter TFLite tidak tersedia di komputer ini, dipakai
     simulasi bobot int8 per-kanal di Keras (dicatat di keluaran).

Keluaran (OUTPUT_DIR):
  - int8_holdout_summary.csv      ringkasan akurasi, ukuran, kesesuaian prediksi
  - int8_holdout_predictions.csv  prediksi per frame (float32 dan int8)
"""
import os, glob, re
import numpy as np
import pandas as pd

# ===================== PENGATURAN =====================
LEARNING_ROOT = r"C:\1. Tukadi_ITS\X_learning5"     # berisi folder Background
HOLDOUT_ROOT = r"C:\1. Tukadi_ITS\X_testing5"       # berisi folder Clean Air dan HFC
CNN_MODEL = r"C:\Users\tukadi\OneDrive\Documents\ChatGPT\New project\ijies_master_outputs\final_model_fixed_epoch.keras"
OUTPUT_DIR = r"C:\1. Tukadi_ITS\hasil_int8"
EXPECTED_FLOAT_ACC = 0.8967     # nilai di naskah, untuk pemeriksaan preprocessing
FRAMES_PER_SESSION = 100
FIRST_HOLDOUT_SESSION = 8       # T8
# ======================================================


def read_frame(path):
    head = open(path).read(2000)
    a = np.loadtxt(path, delimiter=",") if "," in head else np.loadtxt(path)
    return a.reshape(24, 32).astype(np.float32)


def number(p):
    m = re.search(r"\((\d+)\)", os.path.basename(p))
    return int(m.group(1)) if m else 0


def list_csv(folder):
    return sorted(glob.glob(os.path.join(folder, "*.csv")), key=number)


def load_holdout():
    bg = np.mean([read_frame(p) for p in list_csv(os.path.join(LEARNING_ROOT, "Background"))], axis=0)
    X, y, paths, sess = [], [], [], []
    for label, cls in ((0, "Clean Air"), (1, "HFC")):
        for p in list_csv(os.path.join(HOLDOUT_ROOT, cls)):
            X.append(read_frame(p) - bg)
            y.append(label)
            paths.append(os.path.join(cls, os.path.basename(p)))
            sess.append(f"T{(number(p) - 1) // FRAMES_PER_SESSION + FIRST_HOLDOUT_SESSION}")
    return np.asarray(X, np.float32)[..., None], np.asarray(y), paths, sess


def to_labels(out):
    out = np.asarray(out)
    return out.argmax(axis=1) if out.ndim == 2 and out.shape[1] > 1 else (out.ravel() >= 0.5).astype(int)


def tflite_predict(content, X, tf):
    candidates = []
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRT
        candidates.append(("LiteRT", LiteRT))
    except Exception:
        pass
    candidates.append(("tf.lite", tf.lite.Interpreter))
    for name, Interp in candidates:
        try:
            it = Interp(model_content=content)
            it.allocate_tensors()
            i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
            outs = []
            for x in X:
                it.set_tensor(i_in, x[None].astype(np.float32))
                it.invoke()
                outs.append(it.get_tensor(i_out)[0])
            return np.asarray(outs), f"TFLite int8 dynamic-range ({name})"
        except Exception as e:
            print(f"  Interpreter {name} gagal: {type(e).__name__}: {e}")
    return None, None


def simulated_int8(model, X, tf):
    """Simulasi dynamic-range: bobot kernel dikuantisasi int8 simetris per kanal keluaran,
    aktivasi tetap float (prinsip yang sama dengan TFLite dynamic-range quantization)."""
    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    for layer in clone.layers:
        w = layer.get_weights()
        if not w or not hasattr(layer, "kernel"):
            continue
        k = w[0]
        axes = tuple(range(k.ndim - 1))
        scale = np.max(np.abs(k), axis=axes, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        w[0] = np.clip(np.round(k / scale), -127, 127) * scale
        layer.set_weights(w)
    return clone.predict(X, verbose=0), "Simulated int8 weights (per-channel, Keras)"


def main():
    import tensorflow as tf
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y, paths, sess = load_holdout()
    print(f"Holdout: {len(y)} frame ({(y == 0).sum()} Clean Air, {(y == 1).sum()} HFC)")

    model = tf.keras.models.load_model(CNN_MODEL)
    p32 = model.predict(X, verbose=0)
    y32 = to_labels(p32)
    acc32 = (y32 == y).mean()
    print(f"float32 accuracy: {acc32:.4f} ({(y32 == y).sum()}/{len(y)})")
    if abs(acc32 - EXPECTED_FLOAT_ACC) > 0.002:
        print("PERINGATAN: akurasi float32 tidak sama dengan naskah (0.8967). "
              "Periksa preprocessing/path; hasil int8 jangan dipakai sebelum ini cocok.")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl8 = conv.convert()
    p8, method = tflite_predict(tfl8, X, tf)
    if p8 is None:
        p8, method = simulated_int8(model, X, tf)
    y8 = to_labels(p8)
    acc8 = (y8 == y).mean()
    agree = (y8 == y32).mean()
    print(f"int8 accuracy:    {acc8:.4f} ({(y8 == y).sum()}/{len(y)})  [{method}]")
    print(f"prediction agreement float32 vs int8: {agree:.4f}")

    def cm(pred):
        return dict(TN=int(((pred == 0) & (y == 0)).sum()), FP=int(((pred == 1) & (y == 0)).sum()),
                    FN=int(((pred == 0) & (y == 1)).sum()), TP=int(((pred == 1) & (y == 1)).sum()))
    rows = []
    for name, pred in (("float32", y32), ("int8", y8)):
        r = {"model": name, "accuracy": acc32 if name == "float32" else acc8, **cm(pred)}
        for s in sorted(set(sess), key=lambda v: int(v[1:])):
            m = np.array(sess) == s
            r[f"acc_{s}"] = (pred[m] == y[m]).mean()
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ["int8_method"] = ["", method]
    summ["prediction_agreement"] = ["", agree]
    summ["tflite_int8_size_MB"] = ["", len(tfl8) / 1024 / 1024]
    summ.to_csv(os.path.join(OUTPUT_DIR, "int8_holdout_summary.csv"), index=False)
    pd.DataFrame({"path": paths, "session": sess, "true": y, "pred_float32": y32, "pred_int8": y8}).to_csv(
        os.path.join(OUTPUT_DIR, "int8_holdout_predictions.csv"), index=False)
    print("\n", summ.to_string(index=False))
    print("\nSelesai. Hasil tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
