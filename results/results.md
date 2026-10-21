# OmniFM Blockchain Federated Learning — Measured Experimental Results

- **Generated At:** `2026-10-21T08:47:38Z`
- **Hardware:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`PyTorch 2.12.0+cu126`)
- **Client Setup:** 3 sequential logical hospital nodes (`client_1`, `client_2`, `client_3`)
- **Data Integrity Policy:** All values below are directly exported from SQLite `experiments` and `metric_records` tables.

## 1. Experiment Summary Table

| Experiment ID | Task | Algorithm | Status | Rounds | Seed | Peak VRAM (MB) | Final Global Metrics |
|---|---|---|---|---:|---:|---:|---|
| `omnifm_classification_r20_s42` | `classification` | `omnifm` | `COMPLETED` | 20/20 | 42 | 252.30 | `{'train_loss': 0.185807, 'task_loss': 0.185231, 'align_loss': 0.005756, 'accuracy': 0.28125, 'macro_f1': 0.166327}` |
| `omnifm_segmentation_r20_s42` | `segmentation` | `omnifm` | `COMPLETED` | 20/20 | 42 | 342.45 | `{'train_loss': 0.157143, 'task_loss': 0.157122, 'align_loss': 0.000202, 'dice': 0.766211, 'iou': 0.645355}` |
| `omnifm_superresolution_r20_s42` | `superresolution` | `omnifm` | `COMPLETED` | 20/20 | 42 | 310.47 | `{'train_loss': 0.035883, 'task_loss': 0.035854, 'align_loss': 0.000284, 'l1': 0.039861, 'psnr': 26.690152, 'ssim': 0.931677}` |
| `omnifm_fusion_r20_s42` | `fusion` | `omnifm` | `COMPLETED` | 20/20 | 42 | 231.50 | `{'train_loss': 0.124313, 'task_loss': 0.124294, 'align_loss': 0.000187, 'l1': 0.078558, 'spatial_frequency_no_ref': 0.12507, 'ssim': 0.697934, 'vif': 0.619437}` |
| `fedavg_classification_r20_s42` | `classification` | `fedavg` | `COMPLETED` | 20/20 | 42 | 250.17 | `{'train_loss': 0.222022, 'task_loss': 0.222022, 'align_loss': 0.0, 'accuracy': 0.25, 'macro_f1': 0.16259}` |
| `fedprox_classification_r20_s42` | `classification` | `fedprox` | `COMPLETED` | 20/20 | 42 | 307.27 | `{'train_loss': 0.321915, 'task_loss': 0.299683, 'align_loss': 0.0, 'accuracy': 0.15625, 'macro_f1': 0.11004}` |
| `fedper_classification_r20_s42` | `classification` | `fedper` | `COMPLETED` | 20/20 | 42 | 251.12 | `{'train_loss': 0.222022, 'task_loss': 0.222022, 'align_loss': 0.0, 'accuracy': 0.25, 'macro_f1': 0.16259}` |
| `ablation_no_gskr_classification_r20_s42` | `classification` | `omnifm` | `COMPLETED` | 20/20 | 42 | 252.29 | `{'train_loss': 0.374123, 'task_loss': 0.374123, 'align_loss': 0.0, 'accuracy': 0.260417, 'macro_f1': 0.169295}` |
| `ablation_no_eca_classification_r20_s42` | `classification` | `omnifm` | `COMPLETED` | 20/20 | 42 | 252.48 | `{'train_loss': 0.422757, 'task_loss': 0.422155, 'align_loss': 0.006027, 'accuracy': 0.177083, 'macro_f1': 0.088869}` |
| `ablation_no_psp_classification_r20_s42` | `classification` | `omnifm` | `COMPLETED` | 20/20 | 42 | 252.78 | `{'train_loss': 0.25976, 'task_loss': 0.259238, 'align_loss': 0.005211, 'accuracy': 0.40625, 'macro_f1': 0.176433}` |
| `ablation_no_spalign_classification_r20_s42` | `classification` | `omnifm` | `COMPLETED` | 20/20 | 42 | 254.33 | `{'train_loss': 0.349787, 'task_loss': 0.349787, 'align_loss': 0.0, 'accuracy': 0.145833, 'macro_f1': 0.063551}` |
| `omnifm_classification_r80_s42` | `classification` | `omnifm` | `COMPLETED` | 80/80 | 42 | 252.30 | `{'train_loss': 0.025927, 'task_loss': 0.025923, 'align_loss': 4.6e-05, 'accuracy': 0.661458, 'macro_f1': 0.453577}` |
| `omnifm_classification_r100_s42` | `classification` | `omnifm` | `COMPLETED` | 100/100 | 42 | 252.30 | `{'train_loss': 0.029468, 'task_loss': 0.029464, 'align_loss': 4e-05, 'accuracy': 0.641927, 'macro_f1': 0.482692}` |
| `omnifm_segmentation_r100_s42` | `segmentation` | `omnifm` | `COMPLETED` | 100/100 | 42 | 342.70 | `{'train_loss': 0.026862, 'task_loss': 0.02686, 'align_loss': 1.3e-05, 'dice': 0.890259, 'iou': 0.81892}` |
| `omnifm_superresolution_r100_s42` | `superresolution` | `omnifm` | `COMPLETED` | 100/100 | 42 | 310.47 | `{'train_loss': 0.012008, 'task_loss': 0.012007, 'align_loss': 1.7e-05, 'l1': 0.008389, 'psnr': 39.954973, 'ssim': 0.977306}` |
| `omnifm_fusion_r100_s42` | `fusion` | `omnifm` | `COMPLETED` | 100/100 | 42 | 231.50 | `{'train_loss': 0.114836, 'task_loss': 0.114835, 'align_loss': 9e-06, 'l1': 0.074479, 'spatial_frequency_no_ref': 0.126405, 'ssim': 0.704486, 'vif': 0.628657}` |
| `fedavg_classification_r100_s42` | `classification` | `fedavg` | `COMPLETED` | 100/100 | 42 | 250.17 | `{'train_loss': 0.009188, 'task_loss': 0.009188, 'align_loss': 0.0, 'accuracy': 0.617839, 'macro_f1': 0.482035}` |
| `fedprox_classification_r100_s42` | `classification` | `fedprox` | `COMPLETED` | 100/100 | 42 | 307.27 | `{'train_loss': 0.062054, 'task_loss': 0.010536, 'align_loss': 0.0, 'accuracy': 0.654948, 'macro_f1': 0.500914}` |
| `fedper_classification_r100_s42` | `classification` | `fedper` | `COMPLETED` | 100/100 | 42 | 251.12 | `{'train_loss': 0.009188, 'task_loss': 0.009188, 'align_loss': 0.0, 'accuracy': 0.617839, 'macro_f1': 0.482035}` |
| `ablation_no_gskr_classification_r100_s42` | `classification` | `omnifm` | `COMPLETED` | 100/100 | 42 | 252.29 | `{'train_loss': 0.028155, 'task_loss': 0.028155, 'align_loss': 0.0, 'accuracy': 0.658854, 'macro_f1': 0.479002}` |
| `ablation_no_eca_classification_r100_s42` | `classification` | `omnifm` | `COMPLETED` | 100/100 | 42 | 252.48 | `{'train_loss': 0.015036, 'task_loss': 0.015032, 'align_loss': 4.3e-05, 'accuracy': 0.673177, 'macro_f1': 0.494316}` |
| `ablation_no_psp_classification_r100_s42` | `classification` | `omnifm` | `COMPLETED` | 100/100 | 42 | 252.78 | `{'train_loss': 0.020216, 'task_loss': 0.02021, 'align_loss': 6.6e-05, 'accuracy': 0.652995, 'macro_f1': 0.496296}` |
| `ablation_no_spalign_classification_r100_s42` | `classification` | `omnifm` | `COMPLETED` | 100/100 | 42 | 254.33 | `{'train_loss': 0.005074, 'task_loss': 0.005074, 'align_loss': 0.0, 'accuracy': 0.638672, 'macro_f1': 0.498541}` |

## 2. Metric Definitions & Reference Protocols

- **Classification (`dermamnist`, `pathmnist`, `bloodmnist`):** Multi-class `accuracy` and unweighted `macro_f1`. Classifier heads have distinct class semantics (7, 9, and 8 classes) and are kept local (`personalized_parameters`); only the shared ResNet-18 + OmniFM backbone is aggregated.
- **Segmentation (`FeTS2022` T1ce MRI 128×128):** Foreground binary `dice` and `iou` evaluated strictly on labeled validation slices.
- **Super-Resolution (`BreaKHis` 64×64 LR -> x2/x4/x8 HR):** Full-reference `psnr` (dB), `ssim`, and `l1` evaluated strictly on patient-split paired HR test patches.
- **Multimodal Fusion (`CT-MRI`, `PET-MRI`, `SPECT-MRI` 128×128):** Source-pair structural preservation (`ssim`, `vif`, `l1` averaged across both co-registered input modalities) plus no-reference `spatial_frequency_no_ref`.

## 3. Research Limitations

- On-chain SHA-256 hashes and Ganache/Hardhat transactions verify artifact immutability and audit provenance after recording, not cryptographic proof of honest client training.
- Single-GPU sequential client execution simulates multi-hospital federation under a 6 GB VRAM ceiling.
