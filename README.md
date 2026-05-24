# Anomaly Detection in Industrial Sensor Streams

Unsupervised anomaly detection on the **Secure Water Treatment (SWaT)** benchmark using reconstruction-based temporal autoencoders. Models are trained only on normal operation; attacks are flagged when sliding-window reconstruction error exceeds a validation-calibrated threshold.

**Course project** — Deep Learning (time-series unsupervised anomaly detection).

| | |
|---|---|
| **Report** | [`DL_Final_Project_Report.pdf`](DL_Final_Project_Report.pdf) |
| **Code** | [`DL_Final_Project.ipynb`](DL_Final_Project.ipynb) |
| **Repository** | [github.com/rohazubair/DL_Final_Project](https://github.com/rohazubair/DL_Final_Project) |

**Team:** 25280068, 25280010, 25280079

---

## Overview

Industrial cyber-physical systems produce high-frequency multivariate sensor data where labeled attacks are scarce. This project implements an end-to-end pipeline:

1. Domain-aware preprocessing of SWaT telemetry  
2. Chronological train / validation / test splits (normal-only training)  
3. Sliding-window sequence generation  
4. Two autoencoder architectures: **LSTM** (baseline) and **1D convolutional / TCN-style**  
5. Threshold calibration, FPR–TPR analysis, window-size ablations, and latent-space visualization (PCA / t-SNE)

The **TCN autoencoder** achieves the best overall test performance at the default **98.5th-percentile** validation threshold.

| Model | FPR | TPR (Recall) | Precision | F1 |
|-------|-----|--------------|-----------|-----|
| LSTM Autoencoder | 1.47% | 71.12% | 96.81% | **82.00%** |
| TCN Autoencoder | 2.42% | 77.91% | 95.27% | **85.72%** |
| VAE (internal reference) | — | — | — | ≈ 82.8% |

*Full methodology, figures, and ablations are in the PDF report.*

---

## Repository layout

```
DL_Final_Project/
├── DL_Final_Project.ipynb      # Full experiment pipeline (run this)
├── DL_Final_Project_Report.pdf # Written report (abstract, methods, results)
├── data/
│   └── merged.csv              # SWaT dataset (NOT in Git — see below)
├── .gitignore                  # Ignores data/ (large files)
└── README.md
```

---

## Dataset setup

The merged SWaT CSV (~**407 MB**, ~1.4M raw rows) is **not stored in this repository** because it exceeds GitHub’s 100 MB file limit.

1. Obtain `merged.csv` from your course materials or SWaT data source.  
2. Create the folder and place the file:

   ```text
   data/merged.csv
   ```

3. After preprocessing (deduplication, imputation), the notebook uses **946,728** timesteps and **51** sensor/actuator features, consistent with the SWaT literature.

The notebook expects the path `data/merged.csv` (configurable via `MERGED_CSV_PATH` in Section 3).

---

## Requirements

Python 3.9+ recommended. Install dependencies:

```bash
pip install numpy pandas torch matplotlib seaborn scikit-learn jupyter
```

| Package | Use |
|---------|-----|
| `torch` | LSTM / TCN autoencoders, training |
| `pandas`, `numpy` | Data loading and preprocessing |
| `scikit-learn` | `StandardScaler`, PCA, t-SNE |
| `matplotlib`, `seaborn` | Loss curves, threshold plots, latent views |
| `jupyter` | Run the notebook |

GPU is optional; the notebook runs on CPU (`cuda` if available).

---

## How to run

1. Clone the repository and add `data/merged.csv` as described above.  
2. Open and run all cells in order:

   ```bash
   jupyter notebook DL_Final_Project.ipynb
   ```

   Or use JupyterLab / VS Code with the Jupyter extension.

3. **Reproducibility:** seeds are fixed (`torch.manual_seed(42)`, `np.random.seed(42)`).

### Notebook sections

| Section | Content |
|---------|---------|
| 1 | Environment setup and imports |
| 2 | SWaT preprocessing (imputation, label encoding) |
| 3 | Chronological 70/15/15 split and scaling |
| 4 | Sliding windows (default size **64**, stride **16**) |
| 5 | `DeepLSTMAutoencoder`, `DeepTCNAutoencoder`, early stopping |
| 6 | Training loop (AdamW, cosine LR, gradient clipping) |
| 7 | Train both models; benchmark at 98.5% threshold |
| 8 | Multi-threshold FPR / TPR trade-off curves |
| 9 | Window-size ablation {16, 32, 64, 128} on TCN |
| 10 | PCA and t-SNE latent-space plots |
| 11 | Reconstruction error distributions (normal vs attack) |

Training uses **MSE reconstruction loss** on normal windows only; labels are not used during training. Default training: up to **15 epochs**, early stopping (patience 4), batch size **256**, latent dimension **32**.

---

## Method summary

- **Splits:** Chronological 70% train / 15% val / 15% test. Train and validation timesteps are normal-only; the test segment includes attacks (~38.5% attack ratio at timestep level).  
- **Windows:** A window is labeled attack if **any** timestep inside it is an attack.  
- **Detection:** Threshold τ = percentile of validation reconstruction errors (default **98.5%**). Flag test windows with MSE > τ.  
- **Best ablation (TCN):** Window size **128** reaches F1 **86.80%** (see report Table 2).

For equations, architecture diagrams, and figure references, see [`DL_Final_Project_Report.pdf`](DL_Final_Project_Report.pdf).

---

## References

1. Goh et al., *A Dataset to Support Research in the Design of Secure Water Treatment Systems*, CRITIS 2016.  
2. Chalapathy & Chawla, *Deep Learning for Anomaly Detection: A Survey*, arXiv:1901.03407, 2019.  
3. Malhotra et al., *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*, ICML Workshop 2016.  
4. Course project brief: *Anomaly Detection in Industrial Sensor Streams*.

---

## AI tools statement

AI coding assistants were used minimally for notebook debugging, report formatting, and layout. Modeling decisions, experiments, result interpretation, and final writing were completed by the project team (see report for full statement).
