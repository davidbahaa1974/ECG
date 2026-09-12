import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

# Clear any previous plots
plt.close('all')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — set the path to your MIT-BIH record files here
# ═══════════════════════════════════════════════════════════════════
RECORD_NUMBER = "202"  # change this to any record number (100, 101, 102, etc.)
RECORD_PATH  = f"mit-bih-arrhythmia-database-1.0.0/{RECORD_NUMBER}"   # full path
DURATION_SEC = 5       # seconds to analyse
FS           = 360     # sampling frequency (360 Hz for MIT-BIH)
GAIN         = 200     # ADC gain (units/mV)
BASELINE     = 1024    # ADC baseline

print(f"\n>>> Loading record: {RECORD_PATH}\n")


# ═══════════════════════════════════════════════════════════════════
# 1.  Load MIT-BIH .dat
# ═══════════════════════════════════════════════════════════════════
def read_mit_bih_dat(dat_file, n_samples):
    """
    Format 212: every 3 bytes -> 2 x 12-bit signed samples.
      s1 = ((b1 & 0x0F) << 8) | b0
      s2 = (b2 << 4) | ((b1 & 0xF0) >> 4)
    Values >= 2048 are negative (two's complement -> subtract 4096).
    """
    with open(dat_file, "rb") as f:
        raw = f.read()
    ch0, idx = [], 0
    for _ in range(n_samples):
        if idx + 2 >= len(raw):
            break
        b0, b1, b2 = raw[idx], raw[idx + 1], raw[idx + 2]
        idx += 3
        s1 = ((b1 & 0x0F) << 8) | b0
        if s1 >= 2048:
            s1 -= 4096
        ch0.append(s1)
    return np.array(ch0, dtype=float)


# ═══════════════════════════════════════════════════════════════════
# 2.  Filter helpers
# ═══════════════════════════════════════════════════════════════════
def hp_filter(data, cutoff=0.5, fs=360, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, data)

def notch_filter(data, freq=60, fs=360, Q=30):
    b, a = iirnotch(freq / (0.5 * fs), Q)
    return filtfilt(b, a, data)

def lp_filter(data, cutoff=40, fs=360, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, data)


# ═══════════════════════════════════════════════════════════════════
# 3.  Load & convert
# ═══════════════════════════════════════════════════════════════════
n_samples = int(FS * DURATION_SEC)
raw_adc   = read_mit_bih_dat(RECORD_PATH + ".dat", n_samples)
signal    = (raw_adc - BASELINE) / GAIN          # ADC -> mV
time      = np.arange(len(signal)) / FS          # seconds


# ═══════════════════════════════════════════════════════════════════
# 4.  Filter pipeline:  HPF 0.5 Hz -> Notch 50 Hz -> LPF 40 Hz
# ═══════════════════════════════════════════════════════════════════
ecg_filtered = lp_filter(notch_filter(hp_filter(signal)))


# ═══════════════════════════════════════════════════════════════════
# 5.  R-Peak Detection
# ═══════════════════════════════════════════════════════════════════
r_peaks, _ = find_peaks(
    ecg_filtered,
    height   = 0.6 * np.max(ecg_filtered),   # 60% of max amplitude
    distance = int(0.3 * FS),                 # min 300 ms between peaks
)


# ═══════════════════════════════════════════════════════════════════
# 6.  RR Intervals & Heart Rate
# ═══════════════════════════════════════════════════════════════════
rr_intervals = np.diff(r_peaks) / FS
hr_instant   = 60.0 / rr_intervals
hr_mean      = float(np.mean(hr_instant))
hr_min       = float(np.min(hr_instant))
hr_max       = float(np.max(hr_instant))


# ═══════════════════════════════════════════════════════════════════
# 7.  Terminal output
# ═══════════════════════════════════════════════════════════════════
print("=" * 55)
print("            ECG ANALYSIS RESULTS")
print("=" * 55)
print(f"  Record            : {RECORD_PATH}")
print(f"  Sampling frequency: {FS} Hz")
print(f"  Signal duration   : {DURATION_SEC} s")
print(f"  R-peaks detected  : {len(r_peaks)}")
print()
print(f"  {'Beat':<8} {'Sample':>8} {'Time (s)':>10} {'Amp (mV)':>10}")
print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
for i, rp in enumerate(r_peaks):
    print(f"  R{i+1:<7} {rp:>8} {time[rp]:>10.4f} {ecg_filtered[rp]:>10.4f}")
print()
print(f"  {'Interval':<12} {'RR (s)':>8} {'HR (BPM)':>10}")
print(f"  {'-'*12} {'-'*8} {'-'*10}")
for i, (rr, hr) in enumerate(zip(rr_intervals, hr_instant)):
    print(f"  R{i+1}->R{i+2:<7} {rr:>8.4f} {hr:>10.1f}")
print()
print(f"  Average HR : {hr_mean:.1f} BPM")
print(f"  Min HR     : {hr_min:.1f} BPM")
print(f"  Max HR     : {hr_max:.1f} BPM")
print("=" * 55)


# ═══════════════════════════════════════════════════════════════════
# 8.  Plot  — 3 panels
#     [0] Raw ECG
#     [1] Filtered ECG + R-peaks + RR arrows
#     [2] Heart-Rate dashboard (big numbers + beat table)
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#f8f9fb")
fig.suptitle(
    f"ECG Analysis  |  Record {RECORD_PATH}  |  MIT-BIH Arrhythmia Database  "
    f"|  Lead MLII  |  First {DURATION_SEC} s",
    fontsize=13, fontweight="bold", color="#1a1a2e",
)

gs     = fig.add_gridspec(3, 1, height_ratios=[1, 1.3, 1.6], hspace=0.45)
ax_raw = fig.add_subplot(gs[0])
ax_ecg = fig.add_subplot(gs[1], sharex=ax_raw)
ax_hr  = fig.add_subplot(gs[2])

# colours
C_RAW    = "#3a7bd5"
C_FILT   = "#e05c1b"
C_PEAK   = "#d62828"
C_RR     = "#7b2d8b"
C_BG     = "#eef2ff"
C_BORDER = "#3a5fa0"

# ─── Panel 1 : Raw signal ────────────────────────────────────────
ax_raw.set_facecolor("#f0f4ff")
ax_raw.plot(time, signal, color=C_RAW, linewidth=0.8, label="Raw ECG")
ax_raw.set_title("(1) Original ECG Signal  (baseline wander + noise)", fontsize=11)
ax_raw.set_ylabel("Voltage (mV)")
ax_raw.legend(loc="upper right", fontsize=9)
ax_raw.grid(True, alpha=0.3)

# ─── Panel 2 : Filtered + R-peaks ───────────────────────────────
ax_ecg.set_facecolor("#fff7f0")
ax_ecg.plot(time, ecg_filtered, color=C_FILT, linewidth=0.9,
            label="Filtered ECG  (HPF -> Notch -> LPF)", zorder=2)

ax_ecg.scatter(time[r_peaks], ecg_filtered[r_peaks],
               color=C_PEAK, s=110, zorder=5, marker="v",
               label=f"R-peaks  (n = {len(r_peaks)})")

for i, rp in enumerate(r_peaks):
    ax_ecg.axvline(x=time[rp], color=C_PEAK, linestyle="--", alpha=0.35, linewidth=0.9)
    ax_ecg.annotate(
        f"R{i+1}",
        xy=(time[rp], ecg_filtered[rp]),
        xytext=(0, 16), textcoords="offset points",
        ha="center", fontsize=9, fontweight="bold", color="#8b0000",
    )

for i in range(len(r_peaks) - 1):
    t1, t2 = time[r_peaks[i]], time[r_peaks[i + 1]]
    y_arr  = ecg_filtered[r_peaks[i]] - 0.18
    ax_ecg.annotate("", xy=(t2, y_arr), xytext=(t1, y_arr),
                    arrowprops=dict(arrowstyle="<->", color=C_RR, lw=1.3))
    ax_ecg.text((t1 + t2) / 2, y_arr - 0.08,
                f"{rr_intervals[i]:.3f} s\n{hr_instant[i]:.0f} BPM",
                ha="center", fontsize=7.5, color=C_RR)

ax_ecg.set_title(
    "(2) Filtered ECG + R-Peak Detection  (HPF 0.5 Hz -> Notch 60 Hz -> LPF 40 Hz -> find_peaks)",
    fontsize=11)
ax_ecg.set_ylabel("Voltage (mV)")
ax_ecg.set_xlabel("Time (s)")
ax_ecg.legend(loc="upper right", fontsize=9)
ax_ecg.grid(True, alpha=0.3)

# ─── Panel 3 : Heart-Rate Dashboard ─────────────────────────────
ax_hr.set_facecolor(C_BG)
ax_hr.set_xlim(0, 1)
ax_hr.set_ylim(0, 1)
ax_hr.axis("off")

# border
rect = plt.Rectangle((0, 0), 1, 1, linewidth=2,
                      edgecolor=C_BORDER, facecolor="none",
                      transform=ax_hr.transAxes, clip_on=False)
ax_hr.add_patch(rect)

# title banner
ax_hr.text(0.50, 0.96, "(3)  HEART RATE DASHBOARD",
           transform=ax_hr.transAxes, ha="center", va="top",
           fontsize=12, fontweight="bold", color="#1a2e6e",
           bbox=dict(boxstyle="round,pad=0.35", facecolor="#c8d8ff",
                     edgecolor=C_BORDER, linewidth=1.5))

# big average BPM
ax_hr.text(0.50, 0.75, f"{hr_mean:.1f}",
           transform=ax_hr.transAxes, ha="center", va="center",
           fontsize=40, fontweight="bold", color="#c0392b")
ax_hr.text(0.50, 0.63, "BPM  —  Average Heart Rate",
           transform=ax_hr.transAxes, ha="center", va="center",
           fontsize=11, color="#444444")

# min HR (left)
ax_hr.text(0.10, 0.80, f"{hr_min:.1f}",
           transform=ax_hr.transAxes, ha="center", va="center",
           fontsize=28, fontweight="bold", color="#27ae60")
ax_hr.text(0.10, 0.63, "BPM\nMin HR",
           transform=ax_hr.transAxes, ha="center", va="center",
           fontsize=9, color="#555555")

# max HR (right)
ax_hr.text(0.90, 0.80, f"{hr_max:.1f}",
           transform=ax_hr.transAxes, ha="center", va="center",
           fontsize=28, fontweight="bold", color="#e67e22")
ax_hr.text(0.90, 0.63, "BPM\nMax HR",
           transform=ax_hr.transAxes, ha="center", va="center",
           fontsize=9, color="#555555")

# beat-by-beat table header
col_x   = [0.28, 0.40, 0.55, 0.70]
headers = ["Interval", "RR (s)", "HR (BPM)", "Status"]
for cx, hdr in zip(col_x, headers):
    ax_hr.text(cx, 0.54, hdr,
               transform=ax_hr.transAxes, ha="center", va="top",
               fontsize=9, fontweight="bold", color="#1a2e6e")

# horizontal separator line under header
ax_hr.plot([0.24, 0.78], [0.49, 0.49], color="#3a5fa0", linewidth=0.8,
           transform=ax_hr.transAxes, clip_on=False)

# beat-by-beat table rows
row_y = 0.47
for i, (rr, hr) in enumerate(zip(rr_intervals, hr_instant)):
    if hr < 60:
        status, s_color = "Bradycardia", "#2980b9"
    elif hr > 100:
        status, s_color = "Tachycardia", "#c0392b"
    else:
        status, s_color = "Normal",      "#27ae60"

    row_vals   = [f"R{i+1} -> R{i+2}", f"{rr:.4f}", f"{hr:.1f}", status]
    row_colors = ["#333333",            "#333333",   "#c0392b",    s_color]
    for cx, val, fc in zip(col_x, row_vals, row_colors):
        ax_hr.text(cx, row_y, val,
                   transform=ax_hr.transAxes, ha="center", va="top",
                   fontsize=8.5, color=fc)
    row_y -= 0.08

# footer
ax_hr.text(0.50, 0.03,
           f"R-peaks: {len(r_peaks)}   |   Duration: {DURATION_SEC} s"
           f"   |   fs: {FS} Hz   |   Record: {RECORD_PATH}",
           transform=ax_hr.transAxes, ha="center", va="bottom",
           fontsize=8, color="#888888")

# ─── Save & show ─────────────────────────────────────────────────
out_file = "ecg_rpeak_detection.png"   # change path if needed
plt.savefig(out_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved -> {out_file}")