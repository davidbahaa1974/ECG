# ECG R-Peak Detection & Heart Rate Analysis

A Python tool for analyzing ECG signals from the MIT-BIH Arrhythmia Database — detecting R-peaks, computing heart rate, and visualizing results through a clean multi-panel dashboard.

## Overview

This script loads a raw ECG record (format 212, e.g. from the MIT-BIH Arrhythmia Database), applies a filtering pipeline to clean the signal, detects R-peaks, and calculates beat-by-beat heart rate — flagging bradycardia or tachycardia automatically.

## Features

- **MIT-BIH `.dat` file parser** — decodes 12-bit signed samples packed 3 bytes per 2 samples (format 212)
- **Signal filtering pipeline**:
  - High-pass filter (0.5 Hz) — removes baseline wander
  - Notch filter (60 Hz) — removes powerline interference
  - Low-pass filter (40 Hz) — removes high-frequency noise
- **R-peak detection** using `scipy.signal.find_peaks`, with amplitude and minimum-distance thresholds
- **RR interval & heart rate calculation** (instantaneous, average, min, max)
- **Automatic rhythm classification** per interval — Normal / Bradycardia / Tachycardia
- **3-panel visualization**:
  1. Raw ECG signal
  2. Filtered signal with detected R-peaks and RR interval annotations
  3. Heart rate dashboard with summary stats and a beat-by-beat table

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Usage

1. Download a record from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (you need the `.dat` file, e.g. `202.dat`).
2. Place it in a folder named `mit-bih-arrhythmia-database-1.0.0/` alongside the script.
3. Set the record number in the script:

```python
RECORD_NUMBER = "202"  # change to any record number (100, 101, 102, ...)
```

4. Run the script:

```bash
python example.py
```

5. Results print to the terminal and a plot is saved as `ecg_rpeak_detection.png`.

## Configuration

| Parameter        | Default | Description                          |
|-------------------|---------|---------------------------------------|
| `RECORD_NUMBER`    | `"202"` | MIT-BIH record to analyze             |
| `DURATION_SEC`     | `5`     | Seconds of signal to analyze          |
| `FS`               | `360`   | Sampling frequency (Hz)               |
| `GAIN`             | `200`   | ADC gain (units/mV)                   |
| `BASELINE`         | `1024`  | ADC baseline offset                   |

## Output

The script prints:
- Detected R-peak sample indices, timestamps, and amplitudes
- RR intervals and instantaneous heart rate per beat
- Average, minimum, and maximum heart rate

...and saves a labeled dashboard image (`ecg_rpeak_detection.png`) showing the raw signal, filtered signal with R-peaks, and a heart rate summary panel.

## Notes

- Heart rate classification thresholds: **Bradycardia** (< 60 BPM), **Normal** (60–100 BPM), **Tachycardia** (> 100 BPM)
- The `.dat` format decoder is specific to MIT-BIH format 212 (12-bit resolution)
