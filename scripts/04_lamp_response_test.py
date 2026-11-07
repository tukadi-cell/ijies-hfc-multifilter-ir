# -*- coding: utf-8 -*-
"""
Analisis respons MLX90640 terhadap lampu pijar melalui setiap filter optik.

Skrip ini membalik colormap JET yang dipakai saat perekaman
(td_to_image: minTemp = 25, maxTemp = 250) untuk memperkirakan suhu
dari warna video, lalu menghitung:
  - suhu rata-rata area target dan area latar per frame,
  - DeltaT = target - latar (rata-rata dan SD antar-frame),
  - fraksi piksel jenuh (>= batas atas tampilan) di area target.

Keluaran (di folder OUTPUT_DIR):
  - hasil_respons_filter.csv
  - grafik_deltaT_filter.png
  - pratinjau_area_crop.png   (untuk memeriksa posisi area)
"""
import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ===================== PENGATURAN =====================
VIDEO_DIR = r"C:\1. Tukadi_ITS\1. Spectral\video_files"
OUTPUT_DIR = os.path.join(VIDEO_DIR, "hasil_deltaT")

# Skala warna HARUS sama dengan kode rekaman (td_to_image)
MIN_TEMP = 25.0
MAX_TEMP = 250.0

# Nama kondisi -> nama file video (tanpa ekstensi). Sesuaikan bila perlu.
CONDITIONS = {
    "No filter": "No Filter",
    "4.63 µm": "4.63 µm",
    "5.64 µm": "5.64 µm",
    "9.50 µm": "9.50 µm",
    "10.2 µm": "10.2 µm",
}

# Area dalam piksel video 640x480: (x1, y1, x2, y2).
# Nilai awal diperkirakan dari gambar crop_preview; periksa pratinjau_area_crop.png
TARGET_BOX = (228, 228, 328, 329)       # kotak putih (pusat sumber panas)
BACKGROUND_BOX = (20, 49, 171, 169)     # kotak hijau (sudut kiri atas)

SATURATION_LEVEL = 0.98  # indeks warna >= 98% skala dianggap jenuh
# ======================================================


def jet_lut():
    """Tabel warna JET (BGR) untuk indeks 0..255, sama dengan cv2.COLORMAP_JET."""
    idx = np.arange(256, dtype=np.uint8).reshape(256, 1)
    return cv2.applyColorMap(idx, cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)


LUT = jet_lut()


def frame_to_temperature(frame_bgr):
    """Ubah setiap piksel warna menjadi suhu dengan mencari warna JET terdekat."""
    px = frame_bgr.reshape(-1, 3).astype(np.int32)
    d = ((px[:, None, :] - LUT[None, :, :]) ** 2).sum(axis=2)
    index = d.argmin(axis=1)
    temp = MIN_TEMP + index * (MAX_TEMP - MIN_TEMP) / 255.0
    return temp.reshape(frame_bgr.shape[:2]), index.reshape(frame_bgr.shape[:2])


def crop(a, box):
    x1, y1, x2, y2 = box
    return a[y1:y2, x1:x2]


def _norm(s):
    """Samakan variasi penulisan: µ (micro sign) vs μ (huruf Yunani), 'um', spasi, huruf besar."""
    s = s.lower()
    for ext in (".mp4", ".avi"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    for ch in ("µ", "μ"):
        s = s.replace(ch, "u")
    return "".join(c for c in s if c.isalnum() or c == ".")


def find_video(name):
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi"))]
    matches = [f for f in videos if _norm(f) == _norm(name)]
    if not matches:
        raise FileNotFoundError(
            f"Video untuk '{name}' tidak ditemukan di {VIDEO_DIR}.\n"
            f"Video yang tersedia: {videos}")
    return os.path.join(VIDEO_DIR, matches[0])


def analyse_video(path):
    cap = cv2.VideoCapture(path)
    rows, first = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
        if first is None:
            first = frame.copy()
        # Hanya area yang dianalisis yang dikonversi (lebih cepat)
        t_tgt, i_tgt = frame_to_temperature(crop(frame, TARGET_BOX))
        t_bg, _ = frame_to_temperature(crop(frame, BACKGROUND_BOX))
        rows.append({
            "target": t_tgt.mean(),
            "background": t_bg.mean(),
            "saturated_fraction": (i_tgt >= SATURATION_LEVEL * 255).mean(),
        })
    cap.release()
    df = pd.DataFrame(rows)
    df["deltaT"] = df["target"] - df["background"]
    return df, first


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary, previews = [], {}
    for label, name in CONDITIONS.items():
        path = find_video(name)
        df, first = analyse_video(path)
        previews[label] = first
        summary.append({
            "condition": label,
            "video": os.path.basename(path),
            "frames": len(df),
            "target_mean_C": df["target"].mean(),
            "background_mean_C": df["background"].mean(),
            "deltaT_mean_C": df["deltaT"].mean(),
            "deltaT_sd_C": df["deltaT"].std(ddof=1),
            "saturated_fraction": df["saturated_fraction"].mean(),
        })
        print(f"{label:10s} | {len(df):4d} frame | dT = {df['deltaT'].mean():7.1f} "
              f"± {df['deltaT'].std(ddof=1):5.1f} °C | jenuh = {df['saturated_fraction'].mean():.0%}")

    res = pd.DataFrame(summary)
    res.to_csv(os.path.join(OUTPUT_DIR, "hasil_respons_filter.csv"), index=False)

    # ---------- Grafik DeltaT ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(res["condition"], res["deltaT_mean_C"], yerr=res["deltaT_sd_C"],
                  capsize=4, color=colors[:len(res)])
    for b, sat in zip(bars, res["saturated_fraction"]):
        if sat > 0.20:
            b.set_hatch("//")
            b.set_edgecolor("white")
    for b, v in zip(bars, res["deltaT_mean_C"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ΔT = target − background (°C)")
    ax.set_title(f"Camera response through each filter (display range {MIN_TEMP:.0f}–{MAX_TEMP:.0f} °C)\n"
                 "hatched = saturated target (> 20% pixels at display limit; lower bound only)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "grafik_deltaT_filter.png"), dpi=300)

    # ---------- Pratinjau area crop ----------
    fig, axes = plt.subplots(1, len(previews), figsize=(3.2 * len(previews), 2.8))
    for ax, (label, img) in zip(np.atleast_1d(axes), previews.items()):
        img = img.copy()
        cv2.rectangle(img, TARGET_BOX[:2], TARGET_BOX[2:], (255, 255, 255), 2)
        cv2.rectangle(img, BACKGROUND_BOX[:2], BACKGROUND_BOX[2:], (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pratinjau_area_crop.png"), dpi=200)

    make_manuscript_figure(res, previews)
    print("\nSelesai. Hasil tersimpan di:", OUTPUT_DIR)


def make_manuscript_figure(res, previews):
    """Gambar gabungan untuk naskah (bahasa Inggris, tanpa judul internal):
    (a) citra termal tiap kondisi dengan area source dan background,
    (b) grafik DeltaT. Teks overlay Tmin/Tmax di bagian atas frame dipotong."""
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10})
    TOP_CROP = 24  # buang strip teks overlay kamera (Tmin/Tmax/FPS)
    labels = {"No filter": "No filter", "4.63 µm": "4.63 µm", "5.64 µm": "5.64 µm",
              "9.50 µm": "9.50 µm", "10.2 µm": "10.20 µm"}
    n = len(previews)
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 2.6], hspace=0.12, wspace=0.06)

    for k, (label, img) in enumerate(previews.items()):
        ax = fig.add_subplot(gs[0, k])
        img = img.copy()
        cv2.rectangle(img, TARGET_BOX[:2], TARGET_BOX[2:], (255, 255, 255), 3)
        cv2.rectangle(img, BACKGROUND_BOX[:2], BACKGROUND_BOX[2:], (0, 255, 0), 3)
        img = img[TOP_CROP:, :]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(labels.get(label, label), fontsize=9)
        ax.axis("off")
        if k == 0:
            ax.text(-0.08, 1.25, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = fig.add_subplot(gs[1, :])
    names = [labels.get(c, c) for c in res["condition"]]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(names, res["deltaT_mean_C"], yerr=res["deltaT_sd_C"], capsize=4,
                  color=colors[:len(res)], edgecolor="black", linewidth=0.6)
    saturated = res["saturated_fraction"] > 0.20
    for b, sat, v, sd in zip(bars, saturated, res["deltaT_mean_C"], res["deltaT_sd_C"].fillna(0)):
        if sat:
            b.set_hatch("//")
        txt = f"≥ {v:.1f}" if sat else f"{v:.1f}"
        ax.text(b.get_x() + b.get_width() / 2, v + sd + 0.02 * res["deltaT_mean_C"].max(),
                txt, ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ΔT (source − background, °C)")
    ax.set_xlabel("Optical condition")
    ax.set_ylim(0, res["deltaT_mean_C"].max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    if saturated.any():
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="//",
                                 label=f"Saturated at the {MAX_TEMP:.0f} °C display limit (lower bound)")],
                  loc="upper right", fontsize=8, frameon=False)
    ax.text(-0.08, 1.04, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    out = os.path.join(OUTPUT_DIR, "Figure_filter_response.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Gambar naskah:", out)


if __name__ == "__main__":
    main()
